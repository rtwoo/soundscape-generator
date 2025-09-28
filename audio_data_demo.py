#!/usr/bin/env python3
"""
Example demonstrating how to use the audio data storage and retrieval features.

This shows how to:
1. Search for similar audio files
2. Retrieve the actual audio data from the database
3. Save audio files from the database to disk
4. Get information about stored audio files
"""

import os
from pann_audio_embedding import AudioSimilaritySearcher, search_similar_audio

def demo_audio_storage_retrieval():
    """Demo the audio storage and retrieval functionality."""
    
    # Example audio directory (adjust path as needed)
    audio_dir = r"C:\Users\rtwoo\Downloads\FSD50K.dev_audio\FSD50K.dev_audio"
    output_dir = "retrieved_audio"  # Directory to save retrieved audio files
    
    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        print("Please update the audio_dir path in this script.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find some example audio files
    wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]
    if len(wav_files) < 2:
        print("Need at least 2 audio files for demo")
        return
    
    print("=== Audio Storage and Retrieval Demo ===\n")
    
    try:
        # Initialize the searcher
        searcher = AudioSimilaritySearcher()
        
        # 1. Search for similar audio using a query file
        query_file = os.path.join(audio_dir, wav_files[0])
        print(f"1. Searching for audio similar to: {wav_files[0]}")
        
        results = searcher.search_similar_audio(
            query_file, 
            top_k=5, 
            similarity_threshold=0.1  # Low threshold for demo
        )
        
        print(f"Found {len(results)} similar clips:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['fname']} (similarity: {result['similarity']:.3f})")
        
        if not results:
            print("No results found. Make sure the database has been populated with audio data.")
            return
        
        print("\n" + "-" * 50 + "\n")
        
        # 2. Get information about stored audio files
        print("2. Getting information about stored audio files:")
        
        for result in results[:3]:  # Check first 3 results
            fname = result['fname']
            audio_info = searcher.get_audio_info(fname)
            
            if audio_info:
                print(f"File: {fname}")
                print(f"  - Has audio data: {audio_info['has_audio_data']}")
                print(f"  - Size: {audio_info['audio_size_kb']} KB")
                print(f"  - Duration: {audio_info['estimated_duration']}s")
                print()
            else:
                print(f"No info found for {fname}")
        
        print("-" * 50 + "\n")
        
        # 3. Retrieve and save audio data
        print("3. Retrieving and saving audio data:")
        
        for result in results[:2]:  # Save first 2 results
            fname = result['fname']
            output_path = os.path.join(output_dir, f"retrieved_{fname}.wav")
            
            print(f"Retrieving audio for: {fname}")
            
            # Method 1: Get raw audio data
            audio_data = searcher.get_audio_data(fname)
            if audio_data:
                audio_array, sample_rate = audio_data
                print(f"  - Retrieved audio: {len(audio_array)} samples at {sample_rate}Hz")
                print(f"  - Duration: {len(audio_array)/sample_rate:.2f} seconds")
                
                # Method 2: Save directly to file
                success = searcher.save_audio_to_file(fname, output_path)
                if success:
                    print(f"  - Saved to: {output_path}")
                else:
                    print(f"  - Failed to save to: {output_path}")
            else:
                print(f"  - No audio data found for {fname}")
            
            print()
        
        print(f"Retrieved audio files saved to: {output_dir}/")
        
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()

def demo_convenience_function():
    """Demo the convenience function for quick searches."""
    
    audio_dir = r"C:\Users\rtwoo\Downloads\FSD50K.dev_audio\FSD50K.dev_audio"
    
    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        return
    
    wav_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]
    if not wav_files:
        print("No audio files found")
        return
    
    print("=== Convenience Function Demo ===\n")
    
    try:
        # Use the convenience function for quick searches
        query_file = os.path.join(audio_dir, wav_files[1])
        print(f"Quick search for: {wav_files[1]}")
        
        results = search_similar_audio(
            query_file, 
            top_k=3, 
            similarity_threshold=0.2
        )
        
        print(f"Found {len(results)} similar clips:")
        for result in results:
            print(f"  - {result['fname']} (similarity: {result['similarity']:.3f})")
        
    except Exception as e:
        print(f"Error in convenience function demo: {e}")

if __name__ == "__main__":
    print("Choose a demo:")
    print("1. Audio storage and retrieval (full features)")
    print("2. Convenience function (quick search)")
    print("3. Both demos")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        demo_audio_storage_retrieval()
    elif choice == "2":
        demo_convenience_function()
    elif choice == "3":
        demo_audio_storage_retrieval()
        print("\n" + "=" * 60 + "\n")
        demo_convenience_function()
    else:
        print("Invalid choice. Running full demo...")
        demo_audio_storage_retrieval()