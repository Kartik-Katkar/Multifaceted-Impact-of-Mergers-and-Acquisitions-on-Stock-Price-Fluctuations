from IPython.display import display, Image, Audio
import google.generativeai as genai
from pathlib import Path
import requests
import base64
import wave
import time
import json
import cv2
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

# Approximation: 1 second = 30 frames
# Approximation: 1 frame = 760 tokens
# gpt-4-vision-preview limit: 10k tokens
def frame_to_base64(frame):
    _, buffer = cv2.imencode(".png", frame)
    return base64.b64encode(buffer).decode("utf-8")

def video_to_base64(video_path, sample_rate=300):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    base64Frames = []
    with ThreadPoolExecutor() as executor:
        futures = []
        frame_count = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            if frame_count % sample_rate == 0:
                futures.append(executor.submit(frame_to_base64, frame))
            frame_count += 1
        for future in futures:
            base64Frames.append(future.result())

    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames

def generate_script_gemini(base64Frames):
    genai.configure(api_key=gemini_api_key)

    generation_config = {
        "temperature": 0.4,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    model = genai.GenerativeModel(model_name="gemini-pro-vision",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)

    image_parts = [
        {
            "mime_type": "image/png",
            "data": base64_frame
        } for base64_frame in base64Frames
    ]

    prompt_parts = [
        {"text": "These are frames of a video. extract the data from these frames to store in a file. Strictly limit it to 50 words. Note: Only include the narration & do not include timestamp."},
    ] + image_parts

    response = model.generate_content(prompt_parts)
    print(response.text)

    return response.text