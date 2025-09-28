# Frame-to-Audio Similarity API

This FastAPI application processes sequences of JPG frames to generate audio descriptions and find similar audio clips using deep learning models.

## Features

- **Frame Processing**: Accepts multiple JPG frame files via HTTP upload
- **Audio Description**: Uses Qwen2.5-VL-7B-Instruct to generate textual descriptions of audio from frame sequences
- **Audio Synthesis**: Uses AudioLDM2 to synthesize audio from the generated descriptions
- **Similarity Search**: Uses PANN (Pre-trained Audio Neural Networks) embeddings to find similar audio clips from a database
- **Top Results**: Returns the top 3 most similar audio clips with similarity scores

## API Endpoints

### POST `/process_frames`
Upload a sequence of JPG frames and get similar audio clips.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Multiple JPG frame files (field name: `frames`)
- Limits: 1-20 frames per request

**Response:**
```json
{
  "frame_filenames": ["frame1.jpg", "frame2.jpg", "frame3.jpg"],
  "num_frames": 3,
  "audio_description": "sounds of birds chirping and wind rustling through leaves",
  "similar_clips": [
    {
      "fname": "bird_chirping_001",
      "similarity": 0.85,
      "distance": 0.30,
      "split": "dev",
      "metadata": {
        "duration": 5.2,
        "orig_sr": 44100
      }
    }
  ],
  "total_results": 3
}
```

### GET `/health`
Check API health and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "qwen_model": true,
  "qwen_processor": true,
  "audioldm_pipeline": true
}
```

### GET `/`
Simple health check endpoint.

## Setup and Installation

### Prerequisites

1. **Python Environment**: Python 3.8+ with conda/pip
2. **GPU (Recommended)**: CUDA-compatible GPU for faster processing
3. **Database**: PostgreSQL with pgvector extension for similarity search
4. **Models**: The application will download required models on first run

### Installation

1. **Clone/Navigate to the project directory:**
```bash
cd /home/exouser/soundscape-generator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure database connection:**
   - Update database credentials in `pann_search.py` if needed
   - Ensure PostgreSQL with pgvector extension is accessible

### Running the API

#### Option 1: Using the startup script
```bash
./start_api.sh
```

#### Option 2: Direct Python execution
```bash
python fastapi_app.py
```

#### Option 3: Using uvicorn directly
```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Main API**: http://localhost:8000
- **Interactive docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

## Testing

Run the test script to verify everything works:

```bash
python test_api.py
```

This will:
1. Check API health
2. Process the test video (`test2.mp4`)
3. Display results including audio description and similar clips

## Usage Examples

### Using curl
```bash
curl -X POST "http://localhost:8000/process_frames" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "frames=@frame1.jpg" \
     -F "frames=@frame2.jpg" \
     -F "frames=@frame3.jpg" \
     -F "frames=@frame4.jpg"
```

### Using Python requests
```python
import requests

# Prepare multiple frame files
files = []
frame_paths = ['frame1.jpg', 'frame2.jpg', 'frame3.jpg', 'frame4.jpg']
for frame_path in frame_paths:
    files.append(('frames', (frame_path, open(frame_path, 'rb'), 'image/jpeg')))

response = requests.post('http://localhost:8000/process_frames', files=files)

# Close file handles
for _, (_, file_handle, _) in files:
    file_handle.close()

result = response.json()
print(f"Audio description: {result['audio_description']}")
for clip in result['similar_clips']:
    print(f"Similar clip: {clip['fname']} (similarity: {clip['similarity']:.3f})")
```

## Architecture

The API processes frame sequences through the following pipeline:

1. **Frame Upload**: Receives multiple JPG files and saves to temporary locations
2. **Frame Analysis**: Qwen2.5-VL-7B-Instruct analyzes the frame sequence to generate audio description
3. **Audio Synthesis**: AudioLDM2 converts the text description to synthetic audio
4. **Embedding Extraction**: PANN model extracts audio embeddings from synthesized audio
5. **Similarity Search**: PostgreSQL with pgvector performs efficient similarity search
6. **Results**: Returns top 3 similar clips with metadata

## Models Used

- **Qwen2.5-VL-7B-Instruct**: Video-to-text generation for audio descriptions
- **AudioLDM2**: Text-to-audio synthesis
- **PANN (CNN14)**: Audio embedding extraction for similarity search

## Performance Considerations

- **First Request**: Initial model loading takes 2-5 minutes
- **Subsequent Requests**: 30-60 seconds per video (depending on hardware)
- **GPU Acceleration**: Strongly recommended for reasonable performance
- **Memory Requirements**: ~16GB+ RAM, ~8GB+ VRAM recommended

## Configuration

### Model Cache
Models are cached in `/home/exouser/soundscape-generator/` by default. Modify `cache_dir` parameters in `fastapi_app.py` to change location.

### Database Connection
Update database credentials in `pann_search.py`:
```python
DB_HOST = "your-host"
DB_PORT = 5432
DB_NAME = "your-database"
DB_USER = "your-username"
DB_PASS = "your-password"
```

### Audio Parameters
Adjust audio generation parameters in `fastapi_app.py`:
- `audio_length_in_s`: Generated audio length (default: 10 seconds)
- `num_inference_steps`: Audio quality vs speed tradeoff (default: 200)
- `similarity_threshold`: Minimum similarity for results (default: 0.3)

## Error Handling

The API includes comprehensive error handling for:
- Invalid file formats (only MP4 supported)
- Model loading failures
- Database connection issues
- Temporary file cleanup
- Processing timeouts

## Limitations

- **File Format**: Only MP4 videos are supported
- **File Size**: Large videos may cause memory issues
- **Processing Time**: CPU-only processing is very slow
- **Database Dependency**: Requires pre-populated audio database for similarity search

## Troubleshooting

### Common Issues

1. **Models not loading**: Check GPU memory and CUDA installation
2. **Database connection failed**: Verify PostgreSQL and pgvector setup
3. **Out of memory**: Reduce batch sizes or use CPU-only mode
4. **Slow processing**: Ensure GPU acceleration is working
5. **Frame upload errors**: Ensure files are valid JPG format and within size limits
6. **Too many frames**: Maximum 20 frames per request to prevent abuse

### Logs
Check console output for detailed error messages and processing status.

## License

This project uses various open-source models and libraries. Please check individual model licenses for commercial use restrictions.