import json
import os
import numpy as np
from panns_inference import AudioTagging
import psycopg2
from pgvector.psycopg2 import register_vector

# audio IO
import soundfile as sf
import tempfile

from pann_audio_embedding import (
    torch_cuda_available,
    safe_load_audio_from_item,
    load_audio_with_ffmpeg,
    pad_or_skip_audio,
    _PROJ_MATRIX,
    SAMPLE_RATE,
    MIN_AUDIO_LENGTH,
    ORIG_EMBED_DIM
) 

# ---- CONFIG: Edit DB connection ----
DB_HOST = "database-fsd50k.cluster-cry444mo0lui.us-west-1.rds.amazonaws.com"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "rIdDQx7zDwP1GMktXp4f"

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