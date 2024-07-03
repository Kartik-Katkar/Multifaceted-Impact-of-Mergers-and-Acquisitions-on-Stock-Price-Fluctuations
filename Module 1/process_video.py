import os
import whisper
import ffmpeg

video_path = './Test Documents/video with audio.mp4'

if not os.path.exists(video_path):
    raise Exception(f"Video file '{video_path}' does not exist.")

# Function to extract audio from video using FFmpeg
def extract_audio(video_path, audio_file):
    try:
        ffmpeg.input(video_path).output(audio_file, acodec='pcm_s16le').run()
        return True
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return False

# Function to transcribe audio using OpenAI Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

# Extract audio from the video
audio_file = 'temp_audio.wav'
success = extract_audio(video_path, audio_file)

if success:
    print(f"Extracted audio to {audio_file}")

    # Transcribe the audio
    audio_text = transcribe_audio(audio_file)
    os.remove(audio_file)  # Remove temporary audio file

else:
    audio_text = "Audio extraction failed."

output_file = 'data.txt'

with open(output_file, 'a') as f:  # Changed to 'a' to append instead of overwrite
    f.write("Speech-to-Text Transcription:\n")
    f.write(audio_text)
    f.write("\n")

print(f"Transcription appended to '{output_file}' successfully.")
