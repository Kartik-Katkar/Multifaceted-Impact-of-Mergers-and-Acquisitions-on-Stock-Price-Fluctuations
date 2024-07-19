import os
import ffmpeg
import whisper
from process_onlyvideo import video_to_base64,generate_script_gemini
from Ragfilter import classify_and_append

def handle_video(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        if audio_streams:
            print(f'Video file {file_path} has audio.\n')
            audio_file = 'temp_audio.wav'
            success = extract_audio(file_path, audio_file)
            if success:
                audio_text = transcribe_audio(audio_file)
                os.remove(audio_file)  # Remove temporary audio file
                classify_and_append(audio_text)
            else:
                print("Audio extraction failed.\n")
        else:
            base64Frames = video_to_base64(file_path, sample_rate=300)
            if base64Frames:
                content = generate_script_gemini(base64Frames)
                classify_and_append(content)
            else:
                print("Some issue with Gemini.\n")

    except ffmpeg.Error as e:
        print(f'An error occurred while processing the video file {file_path}: {e.stderr.decode("utf-8")}\n')

def extract_audio(video_path, audio_file):
    try:
        ffmpeg.input(video_path).output(audio_file, acodec='pcm_s16le').run()
        return True
    except Exception as e:
        print(f"Error extracting audio: {str(e)}\n")
        return False

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']