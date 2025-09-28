import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="auto",
    attn_implementation="sdpa",
    cache_dir="/home/exouser/soundscape-generator"
)#.half()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir="/home/exouser/soundscape-generator")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "./test2.mp4"},
            {"type": "text", "text": "Describe the sound that would come out of this video."},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    video_fps=2,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
output_text = output_text
print(output_text)

import scipy
import torch
from diffusers import AudioLDM2Pipeline

repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16, cache='/home/exouser/')
pipe = pipe.to(model.device)

# define the prompts
prompt = output_text[0]
negative_prompt = "Low quality."

# set the seed for generator
generator = torch.Generator("cuda").manual_seed(0)

# run the generation
audio = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,
    audio_length_in_s=10.0,
    num_waveforms_per_prompt=3,
    generator=generator,
).audios

# save the best audio sample (index 0) as a .wav file
scipy.io.wavfile.write("test.wav", rate=16000, data=audio[0])
