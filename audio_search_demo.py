#!/usr/bin/env python3
"""
Example usage of the AudioSimilaritySearcher API for finding similar audio clips.

This script demonstrates how external services can use the audio similarity search
functionality to find similar audio clips using HNSW vector search.
"""

import os
import sys
from pann_audio_embedding import AudioSimilaritySearcher

def demo_similarity_search():
    """Demo the audio similarity search functionality."""
    
    # Example audio directory (adjust path as needed)
    audio_dir = r"C:\Users\rtwoo\Downloads\FSD50K.dev_audio\FSD50K.dev_audio"
    
    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        print("Please update the audio_dir path in this script.")
        return
    
    # Find some example audio files
    # wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]
    # if len(wav_files) < 3:
    #     print("Need at least 3 audio files for demo")
    #     return
    
    # # Take first few files as examples
    # example_files = wav_files[:3]
    print("=== Audio Similarity Search Demo ===\n")
    
    # Method 1: Using the convenience function (simpler)
    print("Method 1: Using convenience function")
    print("-" * 40)
    
    query_file = os.path.join(audio_dir, "116090.wav")
    print(f"Query audio: {os.path.basename(query_file)}")

    try:
        searcher = AudioSimilaritySearcher()
        results = searcher.search_similar_audio(
            query_file,
            top_k=5,
            similarity_threshold=0.1  # Low threshold for demo
        )
        
        print(f"Found {len(results)} similar clips:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['fname']}")
            print(f"     Similarity: {result['similarity']:.3f}")
            print(f"     Distance: {result['distance']:.3f}")
            if result['metadata']:
                print(f"     Duration: {result['metadata'].get('duration', 'unknown')}s")
            print()
            
    except Exception as e:
        print(f"Error in similarity search: {e}")
    
    print("\n" + "="*60 + "\n")

def demo_embedding_only():
    """Demo just the embedding functionality without database."""
    
    audio_dir = r"C:\Users\rtwoo\Downloads\FSD50K.dev_audio\FSD50K.dev_audio"
    
    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        return
    
    wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]
    if not wav_files:
        print("No audio files found")
        return
    
    print("=== Audio Embedding Demo ===\n")
    
    try:
        # Initialize searcher (model will be loaded)
        searcher = AudioSimilaritySearcher()
        
        # Embed a few files
        for i, filename in enumerate(wav_files[:3]):
            filepath = os.path.join(audio_dir, filename)
            print(f"Embedding {filename}...")
            
            embedding = searcher.embed_audio_file(filepath)
            
            print(f"  Shape: {embedding.shape}")
            print(f"  Norm: {float(embedding.dot(embedding)**0.5):.6f}")
            print(f"  First 5 values: {embedding[:5]}")
            print()
            
    except Exception as e:
        print(f"Error in embedding demo: {e}")

if __name__ == "__main__":
    print("Choose a demo:")
    print("1. Full similarity search (requires database)")
    print("2. Embedding only (no database required)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        demo_similarity_search()
    elif choice == "2":
        demo_embedding_only()
    else:
        print("Invalid choice. Running embedding demo...")
        demo_embedding_only()