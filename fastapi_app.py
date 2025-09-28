import os
import tempfile
import traceback
from typing import List, Dict, Any

import torch
import scipy.io.wavfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from diffusers import AudioLDM2Pipeline

from pann_search import search_similar_audio

app = FastAPI(
    title="Frame-to-Audio Similarity API",
    description="Upload a sequence of JPG frames to get audio descriptions and find similar audio clips",
    version="1.0.0"
)

# Global variables to store models (loaded once)
qwen_model = None
qwen_processor = None
audioldm_pipe = None

def initialize_models():
    """Initialize all models on startup"""
    global qwen_model, qwen_processor, audioldm_pipe
    
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
    for frame in frames:
        if not frame.filename.lower().endswith(('.jpg', '.jpeg')):
            raise HTTPException(
                status_code=400, 
                detail=f"Only JPG files are supported. Invalid file: {frame.filename}"
            )
    
    temp_frame_paths = []
    temp_audio_path = None
    
    try:
        # Save uploaded frames to temporary files
        print(f"Processing {len(frames)} frames")
        for i, frame in enumerate(frames):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_frame:
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
        similar_clips = search_similar_audio(
            audio_path=temp_audio_path,
            top_k=3,
            similarity_threshold=0.3  # Lower threshold to ensure we get results
        )
        
        print(f"Found {len(similar_clips)} similar clips")
        
        # Format response
        frame_filenames = [frame.filename for frame in frames]
        response = {
            "frame_filenames": frame_filenames,
            "num_frames": len(frames),
            "audio_description": audio_description,
            "similar_clips": similar_clips,
            "total_results": len(similar_clips)
        }
        
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