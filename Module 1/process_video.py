import os
import ffmpeg
import whisper
from process_onlyvideo import video_to_base64,generate_script_gemini

def handle_video(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        if audio_streams:
            append_to_text_file(f'Video file {file_path} has audio.\n')
            audio_file = 'temp_audio.wav'
            success = extract_audio(file_path, audio_file)
            if success:
                audio_text = transcribe_audio(audio_file)
                os.remove(audio_file)  # Remove temporary audio file
                append_to_text_file(audio_text)
            else:
                append_to_text_file("Audio extraction failed.\n")
        else:
            # append_to_text_file(f'Video file {file_path} does not have audio.\n')
            base64Frames = video_to_base64(file_path, sample_rate=300)
            if base64Frames:
                content = generate_script_gemini(base64Frames)
                append_to_text_file(content)
            else:
                append_to_text_file("Some issue with Gemini.\n")

    except ffmpeg.Error as e:
        append_to_text_file(f'An error occurred while processing the video file {file_path}: {e.stderr.decode("utf-8")}\n')

def extract_audio(video_path, audio_file):
    try:
        ffmpeg.input(video_path).output(audio_file, acodec='pcm_s16le').run()
        return True
    except Exception as e:
        append_to_text_file(f"Error extracting audio: {str(e)}\n")
        return False

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

def append_to_text_file(content):
    TEXT_FILE = 'data.txt'
    with open(TEXT_FILE, 'a', encoding='utf-8') as f:
        f.write(content + '\n')