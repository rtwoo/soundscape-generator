

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

# ---- CONFIG: Edit DB connection ----
DB_HOST = "database-fsd50k.cluster-cry444mo0lui.us-west-1.rds.amazonaws.com"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "rIdDQx7zDwP1GMktXp4f"

# Local audio directory configuration
LOCAL_AUDIO_DIR = r"C:\Users\rtwoo\Downloads\FSD50K.dev_audio\FSD50K.dev_audio"

# Hugging Face token + dataset cache (kept for compatibility but not used with local files)
HF_TOKEN = "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"  # set this to avoid rate limits
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

# ============================================================================
# EXTERNAL API FUNCTIONS FOR AUDIO SIMILARITY SEARCH
# ============================================================================

class AudioSimilaritySearcher:
    """
    A class for performing audio similarity search using pre-computed HNSW embeddings.
    This can be used by external services to find similar audio clips.
    """
    
    def __init__(self, db_config=None, model_device=None):
        """
        Initialize the audio similarity searcher.
        
        Args:
            db_config: Dictionary with database connection parameters.
                      If None, uses default config from this module.
            model_device: Device for PANNs model ('cpu', 'cuda'). 
                         If None, auto-detects.
        """
        # Database configuration
        if db_config is None:
            self.db_config = {
                'host': DB_HOST,
                'port': DB_PORT,
                'dbname': DB_NAME,
                'user': DB_USER,
                'password': DB_PASS
            }
        else:
            self.db_config = db_config
        
        # Initialize PANNs model
        device = model_device or ("cuda" if torch_cuda_available() else "cpu")
        print(f"Initializing PANNs model on {device}")
        self.at = AudioTagging(checkpoint_path=None, device=device)
        
        # Pre-computed projection matrix (same as training)
        self._proj_matrix = _PROJ_MATRIX
        
        print("AudioSimilaritySearcher initialized successfully")
    
    def embed_audio_file(self, audio_path):
        """
        Convert an audio file to a normalized embedding vector.
        
        Args:
            audio_path: Path to the audio file (.wav, .mp3, etc.)
            
        Returns:
            numpy.ndarray: Normalized embedding vector of shape (EMBED_DIM,)
            
        Raises:
            RuntimeError: If audio loading or embedding fails
        """
        try:
            # Load and preprocess audio
            arr, sr = safe_load_audio_from_item(audio_path)
            
            # Resample if necessary
            if sr != SAMPLE_RATE:
                try:
                    import librosa
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)
                except ImportError:
                    # Fallback to ffmpeg
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
                        sf.write(tmp_input.name, arr, sr)
                        arr, sr = load_audio_with_ffmpeg(tmp_input.name, target_sr=SAMPLE_RATE)
                        os.unlink(tmp_input.name)
            
            # Handle short audio
            arr_processed, should_skip = pad_or_skip_audio(arr)
            if should_skip:
                raise RuntimeError(f"Audio file too short ({len(arr)/SAMPLE_RATE:.3f}s). Minimum length: {MIN_AUDIO_LENGTH}s")
            
            # Get PANNs embedding
            clipwise_output, embedding = self.at.inference(arr_processed[None, :])
            emb = np.array(embedding)
            
            # Handle different embedding shapes
            if emb.ndim == 3:  # (batch, frames, dim) -> mean pool over frames
                vec = emb.mean(axis=1)[0]
            elif emb.ndim == 2:
                vec = emb[0]
            else:
                raise RuntimeError(f"Unexpected embedding shape: {emb.shape}")
            
            # Ensure correct dimensionality
            if vec.shape[0] != ORIG_EMBED_DIM:
                v = np.zeros(ORIG_EMBED_DIM, dtype=np.float32)
                copy_n = min(vec.shape[0], ORIG_EMBED_DIM)
                v[:copy_n] = vec[:copy_n]
                vec = v
            
            # Apply dimensional reduction
            vec_reduced = np.dot(vec.astype(np.float32), self._proj_matrix)
            
            # Normalize vector
            vec_norm = vec_reduced.astype(np.float32)
            n = np.linalg.norm(vec_norm)
            if n > 0:
                vec_norm = vec_norm / n
            
            return vec_norm
            
        except Exception as e:
            raise RuntimeError(f"Failed to embed audio file {audio_path}: {e}") from e
    
    def search_similar_audio(self, audio_path, top_k=3, similarity_threshold=0.7):
        """
        Find the most similar audio clips to the given audio file.
        
        Args:
            audio_path: Path to the query audio file
            top_k: Number of similar results to return
            similarity_threshold: Minimum cosine similarity (0.0 to 1.0)
            
        Returns:
            list: List of dictionaries with keys:
                - fname: filename of similar audio
                - similarity: cosine similarity score (0.0 to 1.0)
                - metadata: stored metadata as dict
                - distance: L2 distance (lower is more similar)
                
        Raises:
            RuntimeError: If embedding or database query fails
        """
        try:
            # Get embedding for query audio
            query_embedding = self.embed_audio_file(audio_path)
            
            # Connect to database
            conn = psycopg2.connect(**self.db_config)
            register_vector(conn)
            cur = conn.cursor()
            
            # Perform HNSW similarity search using cosine distance
            # Note: pgvector's <=> operator computes cosine distance (0 = identical, 2 = opposite)
            # Cast the list to vector type explicitly
            cur.execute("""
                SELECT fname, split, metadata, embedding, (embedding <=> %s::vector) as distance
                FROM sounds 
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding.tolist(), query_embedding.tolist(), top_k))
            
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            # Convert results to more user-friendly format
            similar_clips = []
            for fname, split, metadata_json, embedding_vec, distance in results:
                # Convert cosine distance to similarity (0=identical, 1=orthogonal, 2=opposite)
                # Similarity = 1 - (distance / 2), clamped to [0, 1]
                similarity = max(0.0, 1.0 - (distance / 2.0))
                
                # Skip results below threshold
                if similarity < similarity_threshold:
                    continue
                
                # Parse metadata
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except:
                    metadata = {}
                
                similar_clips.append({
                    'fname': fname,
                    'similarity': float(similarity),
                    'distance': float(distance),
                    'split': split,
                    'metadata': metadata
                })
            
            return similar_clips
            
        except Exception as e:
            raise RuntimeError(f"Failed to search for similar audio: {e}") from e
    
    def get_audio_data(self, fname):
        """
        Retrieve the stored audio data for a given filename.
        
        Args:
            fname: Filename to retrieve audio data for
            
        Returns:
            tuple: (audio_array, sample_rate) or None if not found
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("SELECT audio_data FROM sounds WHERE fname = %s;", (fname,))
            result = cur.fetchone()
            
            cur.close()
            conn.close()
            
            if result and result[0]:
                # Convert bytes back to audio array
                import io
                audio_bytes = result[0]
                buffer = io.BytesIO(audio_bytes)
                audio_array, sample_rate = sf.read(buffer, dtype='float32')
                buffer.close()
                return audio_array, sample_rate
            else:
                return None
                
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve audio data for {fname}: {e}") from e
    
    def save_audio_to_file(self, fname, output_path):
        """
        Save stored audio data to a file.
        
        Args:
            fname: Filename to retrieve from database
            output_path: Path where to save the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            audio_data = self.get_audio_data(fname)
            if audio_data:
                audio_array, sample_rate = audio_data
                sf.write(output_path, audio_array, sample_rate)
                return True
            else:
                print(f"No audio data found for {fname}")
                return False
                
        except Exception as e:
            print(f"Failed to save audio file: {e}")
            return False
    
    def get_audio_info(self, fname):
        """
        Get information about stored audio without loading the full data.
        
        Args:
            fname: Filename to get info for
            
        Returns:
            dict: Audio information including size, metadata, etc.
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT metadata, 
                       octet_length(audio_data) as audio_size_bytes,
                       CASE WHEN audio_data IS NOT NULL THEN true ELSE false END as has_audio_data
                FROM sounds 
                WHERE fname = %s;
            """, (fname,))
            
            result = cur.fetchone()
            cur.close()
            conn.close()
            
            if result:
                metadata_json, audio_size, has_audio = result
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except:
                    metadata = {}
                
                return {
                    'fname': fname,
                    'metadata': metadata,
                    'audio_size_bytes': audio_size,
                    'audio_size_kb': round(audio_size / 1024, 2) if audio_size else 0,
                    'has_audio_data': has_audio,
                    'estimated_duration': metadata.get('duration', 'unknown')
                }
            else:
                return None
                
        except Exception as e:
            raise RuntimeError(f"Failed to get audio info for {fname}: {e}") from e


def search_similar_audio(audio_path, top_k=3, similarity_threshold=0.5, db_config=None):
    """
    Convenience function to find similar audio clips to a given audio file.
    Creates a searcher instance, embeds the audio, and searches for similar clips.
    
    Args:
        audio_path: Path to the query audio file (.wav, .mp3, etc.)
        top_k: Number of similar results to return (default: 3)
        similarity_threshold: Minimum cosine similarity 0.0-1.0 (default: 0.5)
        db_config: Database configuration dict (default: uses module config)
        
    Returns:
        list: List of similar audio clips with similarity scores
        
    Example:
        results = search_similar_audio("my_audio.wav", top_k=5, similarity_threshold=0.7)
        for result in results:
            print(f"Similar: {result['fname']}, Similarity: {result['similarity']:.3f}")
    """
    searcher = AudioSimilaritySearcher(db_config=db_config)
    return searcher.search_similar_audio(audio_path, top_k, similarity_threshold)

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