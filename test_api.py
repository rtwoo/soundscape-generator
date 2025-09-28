"""
Simple test script for the FastAPI frame processing endpoint.
This script tests the API with sample JPG frames.
"""

import requests
import json
import os
from pathlib import Path
import cv2
import tempfile

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_VIDEO_PATH = "/home/exouser/soundscape-generator/test2.mp4"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the FastAPI server is running.")
        print("   Start it with: python fastapi_app.py")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def extract_frames_from_video(video_path, num_frames=4):
    """Extract frames from video for testing"""
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV not installed. Install with: pip install opencv-python")
        return None
    
    if not os.path.exists(video_path):
        print(f"❌ Test video not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    temp_frame_paths = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_frame_paths.append(temp_file.name)
            cv2.imwrite(temp_file.name, frame)
            temp_file.close()
    
    cap.release()
    return temp_frame_paths

def create_sample_frames():
    """Create simple test frames if video extraction fails"""
    try:
        import numpy as np
        import cv2
    except ImportError:
        print("❌ OpenCV/numpy not available for creating sample frames")
        return None
    
    temp_frame_paths = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Different colors
    
    for i, color in enumerate(colors):
        # Create a simple colored frame
        frame = np.full((240, 320, 3), color, dtype=np.uint8)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_frame_paths.append(temp_file.name)
        cv2.imwrite(temp_file.name, frame)
        temp_file.close()
    
    return temp_frame_paths

def test_frame_processing():
    """Test the main frame processing endpoint"""
    print(f"\nTesting frame processing...")
    
    # Try to extract frames from video first
    temp_frame_paths = extract_frames_from_video(TEST_VIDEO_PATH, num_frames=4)
    
    # Fallback to sample frames if video extraction fails
    if not temp_frame_paths:
        print("Falling back to sample colored frames...")
        temp_frame_paths = create_sample_frames()
    
    if not temp_frame_paths:
        print("❌ Could not create test frames")
        return False
    
    try:
        # Prepare files for upload
        files = []
        for i, frame_path in enumerate(temp_frame_paths):
            files.append(('frames', (f'frame_{i}.jpg', open(frame_path, 'rb'), 'image/jpeg')))
        
        print(f"Uploading {len(files)} frames and processing... (this may take a few minutes)")
        response = requests.post(
            f"{API_BASE_URL}/process_frames",
            files=files,
            timeout=300  # 5 minute timeout
        )
        
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
        
        print(f"Process frames status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Frame processing successful!")
            print(f"Frame filenames: {result.get('frame_filenames')}")
            print(f"Number of frames: {result.get('num_frames')}")
            print(f"Audio description: {result.get('audio_description')}")
            print(f"Number of similar clips found: {result.get('total_results')}")
            
            # Display similar clips
            similar_clips = result.get('similar_clips', [])
            if similar_clips:
                print("\nTop similar audio clips:")
                for i, clip in enumerate(similar_clips, 1):
                    print(f"{i}. {clip.get('fname')} (similarity: {clip.get('similarity', 0):.3f})")
                    metadata = clip.get('metadata', {})
                    if metadata:
                        duration = metadata.get('duration', 'unknown')
                        print(f"   Duration: {duration}s")
            else:
                print("No similar clips found (similarity threshold may be too high)")
            
            return True
        else:
            print(f"❌ Processing failed with status {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Raw response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The frame processing takes time due to model inference.")
        return False
    except Exception as e:
        print(f"❌ Frame processing test failed: {e}")
        return False
    finally:
        # Clean up temporary frame files
        for frame_path in temp_frame_paths:
            try:
                os.unlink(frame_path)
            except:
                pass

def main():
    print("Testing FastAPI Frame-to-Audio Similarity API")
    print("=" * 50)
    
    # Test health check first
    health_ok = test_health_check()
    if not health_ok:
        print("\n❌ Health check failed. Cannot proceed with frame processing test.")
        return
    
    # Test frame processing
    processing_ok = test_frame_processing()
    
    print("\n" + "=" * 50)
    if health_ok and processing_ok:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")

if __name__ == "__main__":
    main()