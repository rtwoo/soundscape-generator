

import argparse
import json
import os
import time
import numpy as np
from tqdm import tqdm
from panns_inference import AudioTagging
import psycopg2
from pgvector.psycopg2 import register_vector

# HF hub helpers (kept for compatibility, but not used with local files)
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub import login as hf_login

import random

# audio IO
import soundfile as sf
import subprocess
import tempfile

import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file

# ---- CONFIG: Edit DB connection ----
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
# Local audio directory configuration
LOCAL_AUDIO_DIR = r"C:\Users\rtwoo\Downloads\FSD50K.dev_audio\FSD50K.dev_audio"

import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file

# Hugging Face token + dataset cache (kept for compatibility but not used with local files)
HF_TOKEN = os.getenv("HF_TOKEN")  # set this to avoid rate limits
CACHE_DIR = os.path.expanduser("~/.cache/hf_datasets")

# PANNs / preprocessing
SAMPLE_RATE = 32000  # PANNs authors used 32k for some configs; panns-inference works with resampled audio
BATCH_SIZE = 8       # batch for embedding extraction (increase for GPU)
MIN_AUDIO_LENGTH = 1.0  # minimum audio length in seconds (PANNs needs at least ~0.5s)
MIN_SAMPLES = int(MIN_AUDIO_LENGTH * SAMPLE_RATE)  # minimum samples needed

# Embedding dims: PANNs (CNN14) original dim, and the reduced dim we want to store
ORIG_EMBED_DIM = 2048
EMBED_DIM = 1024     # target dim to store (<= 2000 so HNSW index works)

# Random projection config (deterministic via seed)
PROJ_SEED = 42
_rng = np.random.default_rng(PROJ_SEED)
_PROJ_MATRIX = (_rng.normal(size=(ORIG_EMBED_DIM, EMBED_DIM)).astype(np.float32)
                / np.sqrt(EMBED_DIM))

# Demo size
DEMO_N = 200

# Download throttling and retry tuning (aggressive to avoid 429)
DOWNLOAD_THROTTLE_SEC = 0.25   # small sleep between download attempts to avoid bursts
MAX_RETRIES = 10
BASE_BACKOFF = 1.0             # seconds, exponential
JITTER = 0.5                   # +/- jitter seconds

def get_local_audio_files(local_dir=LOCAL_AUDIO_DIR, limit=None):
    """
    Scan the local audio directory and return a list of audio file paths.
    
    Args:
        local_dir: Path to the local audio directory
        limit: Optional limit on number of files to return (for demo mode)
        
    Returns:
        List of dictionaries with 'fname' and 'audio_path' keys
    """
    if not os.path.exists(local_dir):
        raise RuntimeError(f"Local audio directory not found: {local_dir}")
    
    audio_files = []
    wav_files = [f for f in os.listdir(local_dir) if f.lower().endswith('.wav')]
    
    # Sort by filename for consistent ordering
    wav_files.sort()
    
    if limit:
        wav_files = wav_files[:limit]
    
    for filename in wav_files:
        audio_path = os.path.join(local_dir, filename)
        # Extract filename without extension for metadata
        fname = os.path.splitext(filename)[0]
        audio_files.append({
            'fname': fname,
            'audio_path': audio_path,
            'file': filename
        })
    
    print(f"Found {len(audio_files)} local audio files in {local_dir}")
    return audio_files

def check_ffmpeg_availability():
    """Check if ffmpeg is available for audio processing fallback."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ ffmpeg is available for audio processing fallback")
            return True
        else:
            print("⚠ ffmpeg command failed - some audio files may not load properly")
            return False
    except FileNotFoundError:
        print("⚠ ffmpeg not found in PATH - install ffmpeg for better audio compatibility")
        print("  Download from: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"⚠ ffmpeg check failed: {e}")
        return False

def ensure_hf_logged_in(token: str = None):
    """
    Log into huggingface_hub programmatically (kept for compatibility but not used with local files).
    """
    print("Using local audio files - Hugging Face login not required.")

# Removed unused HuggingFace helper functions since we're using local files

def load_audio_with_ffmpeg(audio_path, target_sr=SAMPLE_RATE):
    """
    Load and resample audio using ffmpeg directly.
    This provides a robust fallback when soundfile fails.
    
    Args:
        audio_path: Path to the audio file (local or URL)
        target_sr: Target sample rate for resampling
        
    Returns:
        tuple: (numpy_array, sampling_rate) where array is float32
        
    Raises:
        RuntimeError: If ffmpeg fails to process the audio or is not installed
        
    Note:
        Requires ffmpeg to be installed and available in PATH.
        Download from: https://ffmpeg.org/download.html
    """
    try:
        # Create a temporary file for the output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_output = tmp_file.name
        
        # Build ffmpeg command to convert to wav format with target sample rate
        cmd = [
            'ffmpeg',
            '-i', str(audio_path),  # input file
            '-ar', str(target_sr),   # set sample rate
            '-ac', '1',              # convert to mono
            '-f', 'wav',             # output format
            '-y',                    # overwrite output file
            temp_output
        ]
        
        # Run ffmpeg with suppressed output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=30  # 30 second timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with return code {result.returncode}: {result.stderr}")
        
        # Load the converted file with soundfile
        arr, sr = sf.read(temp_output, dtype='float32')
        
        # Clean up temporary file
        os.unlink(temp_output)
        
        return arr, sr
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg timed out after 30 seconds")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure it's in your PATH")
    except Exception as e:
        # Clean up temp file if it exists
        try:
            if 'temp_output' in locals():
                os.unlink(temp_output)
        except:
            pass
        raise RuntimeError(f"ffmpeg audio loading failed: {e}")

def safe_load_audio_from_item(audio_path, retry_attempts=3, base_backoff=0.5):
    """
    Simplified loader for local audio files with ffmpeg fallback.
    
    Args:
        audio_path: Path to the local audio file
        retry_attempts: Number of retry attempts (reduced since we're loading locally)
        base_backoff: Base backoff time for retries
        
    Returns:
        tuple: (numpy_array, sampling_rate)
    """
    if not os.path.exists(audio_path):
        raise RuntimeError(f"Audio file not found: {audio_path}")
    
    # Try soundfile first (fast and reliable for most formats)
    for attempt in range(retry_attempts):
        try:
            arr, sr = sf.read(audio_path, dtype="float32")
            return arr, sr
        except Exception as e:
            print(f"soundfile failed for {audio_path} (attempt {attempt+1}/{retry_attempts}): {e}")
            if attempt < retry_attempts - 1:
                time.sleep(base_backoff * (attempt + 1))
    
    # Fallback to ffmpeg if soundfile fails
    print(f"Attempting ffmpeg fallback for {audio_path}")
    try:
        arr, sr = load_audio_with_ffmpeg(audio_path, target_sr=SAMPLE_RATE)
        return arr, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {audio_path} with both soundfile and ffmpeg. Last error: {e}")

def connect_db():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    cur = conn.cursor()
    try:
        # Try to create the extension (harmless if already present)
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    except psycopg2.ProgrammingError as e:
        # Could be permission error or extension unavailable on managed DB
        conn.rollback()
        print("Warning: could not CREATE EXTENSION vector. Error:", e)
        print("If you are on RDS/Aurora, ensure your Postgres engine version supports pgvector.")
    finally:
        cur.close()

    # Now register the vector type with psycopg2 (requires the SQL type to exist)
    try:
        register_vector(conn)
    except Exception as e:
        conn.close()
        raise RuntimeError(
            "pgvector type not available. Ensure the 'vector' extension exists in the "
            "database and your DB user has permission to create extensions. "
            f"Underlying error: {e}"
        ) from e

    return conn

def ensure_table(conn):
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sounds (
      id bigserial PRIMARY KEY,
      fname text NOT NULL,
      split text,
      metadata jsonb,
      embedding vector(%s),
      audio_data bytea
    );
    """ % EMBED_DIM)
    
    # Add the audio_data column if it doesn't exist (for existing databases)
    cur.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                      WHERE table_name='sounds' AND column_name='audio_data') THEN
            ALTER TABLE sounds ADD COLUMN audio_data bytea;
        END IF;
    END$$;
    """)
    # create HNSW index (fast & good tradeoff) - requires pgvector support in RDS
    cur.execute("""
    DO $$
    BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE tablename='sounds' AND indexname='sounds_embedding_hnsw'
    ) THEN
        -- Use vector_cosine_ops if your vectors are normalized (recommended).
        CREATE INDEX sounds_embedding_hnsw ON sounds USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
    END IF;
    END$$;
    """)
    conn.commit()
    cur.close()

def insert_batch(conn, rows):
    """
    rows: list of tuples (fname, split, metadata_json, vector_list, audio_data_bytes)
    """
    cur = conn.cursor()
    # mogrify each row; careful about size for very large batches
    args_str = ",".join(cur.mogrify("(%s,%s,%s,%s,%s)", r).decode("utf8") for r in rows)
    cur.execute("INSERT INTO sounds (fname, split, metadata, embedding, audio_data) VALUES " + args_str + " RETURNING id;")
    ids = [r[0] for r in cur.fetchall()]
    conn.commit()
    cur.close()
    return ids

def norm_vec(v: np.ndarray) -> list:
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v.tolist()

def audio_to_bytes(audio_array, sample_rate=SAMPLE_RATE):
    """
    Convert audio array to WAV format bytes for database storage.
    
    Args:
        audio_array: numpy array of audio samples (float32)
        sample_rate: sample rate of the audio
        
    Returns:
        bytes: WAV format audio data
    """
    try:
        import io
        
        # Create a BytesIO buffer to hold the WAV data
        buffer = io.BytesIO()
        
        # Write audio to buffer in WAV format
        sf.write(buffer, audio_array, sample_rate, format='WAV', subtype='PCM_16')
        
        # Get the bytes
        audio_bytes = buffer.getvalue()
        buffer.close()
        
        return audio_bytes
        
    except Exception as e:
        print(f"Warning: Failed to convert audio to bytes: {e}")
        return None

def pad_or_skip_audio(arr, min_samples=MIN_SAMPLES):
    """
    Handle audio that's too short for PANNs processing.
    
    Args:
        arr: audio array
        min_samples: minimum number of samples required
        
    Returns:
        tuple: (processed_array, should_skip)
            - processed_array: padded audio or None if should skip
            - should_skip: True if audio should be skipped
    """
    if len(arr) < min_samples:
        if len(arr) < min_samples // 4:  # If extremely short (< 0.25s), skip
            return None, True
        
        # Pad short audio by repeating it or zero-padding
        if len(arr) > min_samples // 8:  # If at least 0.125s, repeat the audio
            repeat_count = (min_samples // len(arr)) + 1
            arr_repeated = np.tile(arr, repeat_count)
            arr = arr_repeated[:min_samples]
        else:  # Very short, zero-pad
            arr_padded = np.zeros(min_samples, dtype=arr.dtype)
            arr_padded[:len(arr)] = arr
            arr = arr_padded
    
    return arr, False



def run(mode="demo", local_dir=LOCAL_AUDIO_DIR):
    # Load panns model (CNN14)
    device = "cuda" if (torch_cuda_available()) else "cpu"
    at = AudioTagging(checkpoint_path=None, device=device)  # will download checkpoints if needed

    # Connect DB
    conn = connect_db()
    ensure_table(conn)

    # Load audio files from local directory instead of HuggingFace
    print(f"Loading audio files from local directory: {local_dir}")
    
    if mode == "full":
        # Load all available local files
        audio_files = get_local_audio_files(local_dir)
        iterator = audio_files
        total = len(audio_files)
    else:
        # Load limited number for demo
        audio_files = get_local_audio_files(local_dir, limit=DEMO_N)
        iterator = audio_files
        total = len(iterator)
    
    print(f"Processing {total} audio files in {mode} mode")

    # Build batches of raw numpy arrays for panns_inference
    batch_audio = []
    batch_meta = []
    inserted = 0

    # Unified processing loop for both demo and full modes
    prog = tqdm(iterator, total=total, unit="clips")
    for item in prog:
        audio_path = item.get("audio_path")
        fname = item.get("fname", "unknown")
        
        try:
            arr, sr = safe_load_audio_from_item(audio_path)
        except Exception as e:
            print(f"Skipping clip {fname} due to load failure: {e}")
            continue

        # resample if necessary to SAMPLE_RATE
        if sr != SAMPLE_RATE:
            try:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)
            except ImportError:
                print(f"librosa not available, using ffmpeg for resampling from {sr}Hz to {SAMPLE_RATE}Hz")
                # Create temp file with current audio data
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
                    sf.write(tmp_input.name, arr, sr)
                    arr, sr = load_audio_with_ffmpeg(tmp_input.name, target_sr=SAMPLE_RATE)
                    os.unlink(tmp_input.name)
        
        # Handle short audio files that would cause PANNs to fail
        arr_processed, should_skip = pad_or_skip_audio(arr)
        if should_skip:
            print(f"Skipping {fname}: audio too short ({len(arr)/SAMPLE_RATE:.3f}s)")
            continue
        
        # Store both original audio (for database) and processed audio (for PANNs)
        batch_audio.append((arr, arr_processed))  # (original, processed)
        batch_meta.append((fname, "dev", {"orig_sr": sr, "local_path": audio_path, "duration": len(arr)/SAMPLE_RATE}))
        
        if len(batch_audio) >= BATCH_SIZE:
            process_and_store(at, conn, batch_audio, batch_meta)
            inserted += len(batch_audio)
            batch_audio, batch_meta = [], []
    
    # Process remaining files
    if batch_audio:
        process_and_store(at, conn, batch_audio, batch_meta)
        inserted += len(batch_audio)
    
    prog.close()

    print("Done. Inserted rows:", inserted)
    conn.close()

def process_and_store(at, conn, audio_list, meta_list):
    """
    audio_list: list of tuples (original_audio, processed_audio) - both are 1D numpy arrays (float32), resampled to SAMPLE_RATE
    meta_list: list of tuples (fname, split, metadata_dict)
    """
    rows = []
    for (arr_original, arr_processed), (fname, split, meta) in zip(audio_list, meta_list):
        try:
            # Use processed audio for PANNs inference (may be padded/repeated)
            clipwise_output, embedding = at.inference(arr_processed[None, :])  # arr_processed[None,:] -> (1, samples)
            emb = np.array(embedding)
            if emb.ndim == 3:  # (batch, frames, dim) -> mean pool over frames
                vec = emb.mean(axis=1)[0]
            elif emb.ndim == 2:
                vec = emb[0]
            else:
                raise RuntimeError("unexpected embedding shape: " + str(emb.shape))

            # --- reduce dimensionality via deterministic Gaussian random projection ---
            # vec shape expected (ORIG_EMBED_DIM,)
            if vec.shape[0] != ORIG_EMBED_DIM:
                # If the PANNs wrapper changes the dimension, try to handle gracefully
                v = np.zeros(ORIG_EMBED_DIM, dtype=np.float32)
                copy_n = min(vec.shape[0], ORIG_EMBED_DIM)
                v[:copy_n] = vec[:copy_n]
                vec = v
            # Project to target dim
            vec_reduced = np.dot(vec.astype(np.float32), _PROJ_MATRIX)  # shape (EMBED_DIM,)
            # Normalize (recommended when using cosine operator)
            vec_norm = norm_vec(vec_reduced)
            
            # Convert ORIGINAL audio to bytes for storage (not the processed/padded version)
            audio_bytes = audio_to_bytes(arr_original, SAMPLE_RATE)
            
            rows.append((fname, split, json.dumps(meta), vec_norm, audio_bytes))
        except Exception as e:
            duration = meta.get("duration", "unknown")
            print(f"PANNs inference failed for {fname} (duration: {duration}s): {e}")
            continue
    
    # insert all valid rows
    if rows:
        insert_batch(conn, rows)

# small helper to detect torch cuda existence without importing heavy torch top-level repeatedly
def torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    parser.add_argument("--local-dir", default=LOCAL_AUDIO_DIR, 
                       help="Path to local audio directory (default: %(default)s)")
    args = parser.parse_args()

    # Check audio processing dependencies
    check_ffmpeg_availability()
    
    # Run the main processing with the specified directory
    run(mode=args.mode, local_dir=args.local_dir)