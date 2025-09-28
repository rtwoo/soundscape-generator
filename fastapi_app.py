import os
import io
import base64
import tempfile
import traceback
from typing import List, Dict, Any

import numpy as np
import torch
import scipy.io.wavfile
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from diffusers import AudioLDM2Pipeline

from pann_search import AudioSimilaritySearcher

app = FastAPI(
    title="Frame-to-Audio Similarity API",
    description="Upload a sequence of JPG frames to get audio descriptions and find similar audio clips",
    version="1.0.0"
)

# Global variables to store models (loaded once)
qwen_model = None
qwen_processor = None
audioldm_pipe = None
audio_searcher = None

def initialize_models():
    """Initialize all models on startup"""
    global qwen_model, qwen_processor, audioldm_pipe, audio_searcher
    
    print("Loading Qwen2.5-VL model...")
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map="auto",
        attn_implementation="sdpa",
        cache_dir="/media/volume/sunhacks"
    )
    
    qwen_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        cache_dir="/media/volume/sunhacks"
    )
    
    print("Loading AudioLDM2 model...")
    audioldm_pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2", 
        torch_dtype=torch.float16, 
        cache='/media/volume/sunhacks'
    )
    audioldm_pipe = audioldm_pipe.to(qwen_model.device)
    
    print("Models loaded successfully!")

    ensure_audio_searcher()


def ensure_audio_searcher():
    global audio_searcher
    if audio_searcher is None:
        print("Initializing audio similarity searcher...")
        audio_searcher = AudioSimilaritySearcher()
        print("Audio similarity searcher ready!")

@app.on_event("startup")
async def startup_event():
    """Initialize models when the API starts"""
    initialize_models()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Video-to-Audio Similarity API is running"}

@app.post("/process_frames")
async def process_frames(frames: List[UploadFile] = File(...)):
    """
    Upload a sequence of JPG frames and get similar audio clips.
    
    This endpoint:
    1. Receives a sequence of JPG frame files
    2. Uses Qwen2.5-VL to generate audio description from the frame sequence
    3. Uses AudioLDM2 to synthesize audio from description
    4. Searches for similar audio clips using PANN embeddings
    5. Returns the top 3 most similar audio clips
    """
    
    # Validate number of frames
    if not frames or len(frames) == 0:
        raise HTTPException(
            status_code=400, 
            detail="At least one frame must be provided"
        )
    
    if len(frames) > 20:  # Reasonable limit to prevent abuse
        raise HTTPException(
            status_code=400, 
            detail="Maximum 20 frames allowed"
        )
    
    # Validate file types
    # for frame in frames:
    #     if not frame.filename.lower().endswith(('.jpg', '.jpeg')):
    #         raise HTTPException(
    #             status_code=400, 
    #             detail=f"Only JPG files are supported. Invalid file: {frame.filename}"
    #         )
    
    temp_frame_paths = []
    temp_audio_path = None
    
    try:
        # Save uploaded frames to temporary files
        print(f"Processing {len(frames)} frames")
        for i, frame in enumerate(frames):
            original_ext = os.path.splitext(frame.filename or "frame.jpg")[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as temp_frame:
                temp_frame_path = temp_frame.name
                temp_frame_paths.append(temp_frame_path)
                content = await frame.read()
                temp_frame.write(content)
                print(f"Saved frame {i+1}: {frame.filename}")
        
        # Step 1: Generate audio description using Qwen2.5-VL
        print("Generating audio description from frame sequence...")
        
        # Use direct file paths instead of file:// URIs for QwenVL
        # The model expects local file paths, HTTP URLs, or base64 strings, not file:// scheme
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": temp_frame_paths,  # Pass direct file paths
                    },
                    {"type": "text", "text": "Describe the sound that would come out of this video."},
                ],
            }
        ]
        
        # Process vision info from the original conversation structure (not the template)
        image_inputs, video_inputs, video_kwargs = process_vision_info(conversation, return_video_kwargs=True)
        
        # Get the text template for tokenization
        text = qwen_processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Now tokenize and create inputs
        inputs = qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")
        
        # Generate description
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        audio_description = output_text[0] if output_text else "ambient sound"
        print(f"Generated description: {audio_description}")
        
        # Step 2: Generate synthetic audio using AudioLDM2
        print("Synthesizing audio from description...")
        negative_prompt = "Low quality."
        generator = torch.Generator(qwen_model.device.type).manual_seed(0)
        
        audio = audioldm_pipe(
            audio_description,
            negative_prompt=negative_prompt,
            num_inference_steps=200,
            audio_length_in_s=10.0,
            num_waveforms_per_prompt=1,  # Generate only 1 for efficiency
            generator=generator,
        ).audios
        
        # Save synthesized audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio_path = temp_audio.name
            scipy.io.wavfile.write(temp_audio_path, rate=16000, data=audio[0])
        
        print("Audio synthesized successfully")
        
        # Step 3: Search for similar audio clips using PANN embeddings
        print("Searching for similar audio clips...")

        if audio_searcher is None:
            print("Audio searcher not yet initialized. Initializing now...")
            ensure_audio_searcher()

        similar_clips = audio_searcher.search_similar_audio(
            audio_path=temp_audio_path,
            top_k=3,
            similarity_threshold=0.3  # Lower threshold to ensure we get results
        )

        print(f"Found {len(similar_clips)} similar clips")

        # Encode generated audio
        generated_audio_payload = None
        if audio.any():
            try:
                print(f"Encoding generated audio: shape={audio[0].shape}")
                generated_buffer = io.BytesIO()
                scipy.io.wavfile.write(generated_buffer, rate=16000, data=audio[0])
                generated_buffer.seek(0)
                generated_bytes = generated_buffer.read()
                
                generated_audio_payload = {
                    "filename": "generated_soundscape.wav",
                    "sample_rate": 16000,
                    "content_base64": base64.b64encode(generated_bytes).decode("utf-8"),
                }
                print(f"Successfully encoded generated audio ({len(generated_bytes)} bytes)")
                
            except Exception as gen_audio_error:
                print(f"Error encoding generated audio: {gen_audio_error}")
                import traceback
                traceback.print_exc()
        else:
            print("No generated audio available")

        # Collect similar clip payloads with audio bytes
        similar_clip_payloads = []
        for i, clip in enumerate(similar_clips):
            print(f"Processing similar clip {i+1}/{len(similar_clips)}: {clip.get('fname')}")
            
            clip_payload = {
                "fname": clip.get("fname"),
                "similarity": clip.get("similarity"),
                "metadata": clip.get("metadata", {}),
                "split": clip.get("split"),
            }

            # Try to retrieve audio data
            audio_tuple = None
            if audio_searcher is not None and clip.get("fname"):
                try:
                    print(f"Attempting to retrieve audio data for {clip['fname']}")
                    
                    # First, check if the audio info exists
                    audio_info = audio_searcher.get_audio_info(clip["fname"])
                    if audio_info:
                        print(f"Audio info for {clip['fname']}: {audio_info}")
                    else:
                        print(f"No audio info found for {clip['fname']}")
                    
                    # Try to get the actual audio data
                    audio_tuple = audio_searcher.get_audio_data(clip["fname"])
                    if audio_tuple:
                        print(f"Successfully retrieved audio data for {clip['fname']}")
                    else:
                        print(f"No audio data found for {clip['fname']} - may not be stored in database")
                        
                except Exception as retrieval_error:
                    print(f"Error retrieving audio data for {clip.get('fname')}: {retrieval_error}")
                    import traceback
                    traceback.print_exc()

            if audio_tuple:
                try:
                    audio_array, sample_rate = audio_tuple
                    print(f"Audio data: shape={audio_array.shape}, sample_rate={sample_rate}")
                    
                    buffer = io.BytesIO()
                    sf.write(buffer, audio_array, sample_rate, format="WAV")
                    buffer.seek(0)
                    audio_bytes = buffer.read()
                    
                    clip_payload.update({
                        "sample_rate": sample_rate,
                        "content_base64": base64.b64encode(audio_bytes).decode("utf-8"),
                    })
                    print(f"Successfully encoded audio for {clip['fname']} ({len(audio_bytes)} bytes)")
                    
                except Exception as encoding_error:
                    print(f"Error encoding audio for {clip['fname']}: {encoding_error}")
                    import traceback
                    traceback.print_exc()
                    clip_payload.update({"sample_rate": None, "content_base64": None})
            else:
                print(f"No audio data available for {clip['fname']} - creating placeholder")
                
                # Create a short placeholder audio (1 second of silence)
                try:
                    placeholder_duration = 1.0  # seconds
                    placeholder_sample_rate = 16000
                    placeholder_samples = int(placeholder_duration * placeholder_sample_rate)
                    placeholder_audio = np.zeros(placeholder_samples, dtype=np.float32)
                    
                    # Add a small beep to indicate this is a placeholder
                    beep_freq = 440  # A4 note
                    beep_duration = 0.1  # 100ms beep
                    beep_samples = int(beep_duration * placeholder_sample_rate)
                    t = np.linspace(0, beep_duration, beep_samples)
                    beep = 0.1 * np.sin(2 * np.pi * beep_freq * t)
                    placeholder_audio[:beep_samples] = beep
                    
                    buffer = io.BytesIO()
                    sf.write(buffer, placeholder_audio, placeholder_sample_rate, format="WAV")
                    buffer.seek(0)
                    placeholder_bytes = buffer.read()
                    
                    clip_payload.update({
                        "sample_rate": placeholder_sample_rate,
                        "content_base64": base64.b64encode(placeholder_bytes).decode("utf-8"),
                        "is_placeholder": True,  # Flag to indicate this is a placeholder
                    })
                    print(f"Created placeholder audio for {clip['fname']} ({len(placeholder_bytes)} bytes)")
                    
                except Exception as placeholder_error:
                    print(f"Error creating placeholder audio for {clip['fname']}: {placeholder_error}")
                    clip_payload.update({"sample_rate": None, "content_base64": None})

            similar_clip_payloads.append(clip_payload)

        # Format response
        frame_filenames = [frame.filename for frame in frames]
        response = {
            "frame_filenames": frame_filenames,
            "num_frames": len(frames),
            "audio_description": audio_description,
            "generated_audio": generated_audio_payload,
            "similar_clips": similar_clip_payloads,
            "total_results": len(similar_clip_payloads)
        }
        print(response)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )
    
    finally:
        # Clean up temporary files
        for temp_frame_path in temp_frame_paths:
            if os.path.exists(temp_frame_path):
                try:
                    os.unlink(temp_frame_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp frame file: {e}")
        
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                print(f"Warning: Could not delete temp audio file: {e}")

@app.get("/health")
async def health_check():
    """Detailed health check with model status"""
    try:
        models_loaded = all([
            qwen_model is not None,
            qwen_processor is not None,
            audioldm_pipe is not None
        ])
        
        return {
            "status": "healthy" if models_loaded else "initializing",
            "models_loaded": models_loaded,
            "qwen_model": qwen_model is not None,
            "qwen_processor": qwen_processor is not None,
            "audioldm_pipeline": audioldm_pipe is not None
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)