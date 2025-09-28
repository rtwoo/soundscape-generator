# index_sounds_to_rds.py
"""
Requirements:
  pip install panns-inference librosa numpy soundfile psycopg2-binary pgvector
Run:
  python index_sounds_to_rds.py /path/to/sounds_dir
"""

import os
import sys
import json
import numpy as np
import librosa
import psycopg2
from pgvector.psycopg2 import register_vector
from panns_inference import AudioTagging  # panns-inference wrapper

# --- Config: edit for your RDS instance ---
DB_HOST = "your-rds-host.amazonaws.com"
DB_PORT = 5432
DB_NAME = "your_db"
DB_USER = "master_user"
DB_PASS = "master_password"

SAMPLE_RATE = 32000   # panns authors used 32k but 16k is also common; CNN14 supports these. See paper.
BATCH_DEVICE = "cpu"  # 'cuda' if you have GPU

# --- helpers ---
def load_audio(path, sr=SAMPLE_RATE):
    # librosa loads as float32 mono
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio

def connect_db():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    # register pgvector types on this connection so psycopg2 will send/receive vectors
    register_vector(conn)
    return conn

def ensure_table(conn):
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sounds (
      id bigserial PRIMARY KEY,
      filename text NOT NULL,
      metadata jsonb,
      embedding vector(2048)
    );
    """)
    conn.commit()
    cur.close()

# --- main indexing pipeline ---
def main(sounds_dir):
    # init PANNs audio tagger (CNN14). checkpoint_path=None uses default checkpoint download path.
    at = AudioTagging(checkpoint_path=None, device=BATCH_DEVICE)

    conn = connect_db()
    ensure_table(conn)
    cur = conn.cursor()

    files = [f for f in os.listdir(sounds_dir) if f.lower().endswith(('.wav','.flac','.mp3'))]
    for fname in files:
        path = os.path.join(sounds_dir, fname)
        audio = load_audio(path)            # shape (n_samples,)
        audio_batch = audio[None, :]        # (batch, samples) as expected by panns_inference

        # inference: returns (clipwise_output, embedding)
        # embedding shape often (batch, 2048) or (batch, T, 2048) depending on model config.
        clipwise_output, embedding = at.inference(audio_batch)
        # if embedding is sequence, pool to a single vector (mean)
        emb_arr = np.array(embedding)
        if emb_arr.ndim == 3:
            # (batch, frames, dim)
            vec = emb_arr.mean(axis=1)[0]
        elif emb_arr.ndim == 2:
            # (batch, dim)
            vec = emb_arr[0]
        else:
            raise RuntimeError("Unexpected embedding shape: " + str(emb_arr.shape))

        # normalize (good practice for cosine or L2 comparisons)
        vec = vec.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = (vec / norm).tolist()
        else:
            vec = vec.tolist()

        metadata = {"sample_rate": SAMPLE_RATE, "source": "my_library"}
        cur.execute(
            "INSERT INTO sounds (filename, metadata, embedding) VALUES (%s, %s, %s) RETURNING id",
            (fname, json.dumps(metadata), vec)
        )
        inserted_id = cur.fetchone()[0]
        conn.commit()
        print(f"Inserted {fname} as id={inserted_id}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python index_sounds_to_rds.py /path/to/sounds_dir")
        sys.exit(1)
    main(sys.argv[1])
