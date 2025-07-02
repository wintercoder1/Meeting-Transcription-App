# transcription_worker.py
import whisper
import sounddevice as sd
import numpy as np
import queue
from dotenv import load_dotenv
import threading
import time
import os
from Constants import ENV_FILE
from datetime import datetime
from openai import OpenAI

model = whisper.load_model("base")
samplerate = 16000
blocksize = 4000
audio_q = queue.Queue()

PAUSE_SECONDS = 4
CHUNK_TIME_SECONDS = 50

# Shared buffer state
buffer = []
last_caption_time = time.time()
last_summary_time = time.time()

client = None  # <- initialize later


def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status)
    audio_q.put(indata.copy())

def summarize_buffer():
    global buffer, last_summary_time
    if not buffer:
        return None

    full_text = "\n".join(buffer).strip()
    if not full_text:
        return None

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": f"Summarize the following meeting transcript chunk:\n\n{full_text}"
        }],
        temperature=0.3
    )
    summary = response.choices[0].message.content
    buffer.clear()
    last_summary_time = time.time()
    return summary

def transcribe_stream(transcript_callback, summary_callback, should_continue, is_auto_summary_enabled, get_summary_interval_sec):
    global last_caption_time
    print("🎧 Listening and transcribing...")
    temp_buffer = []

    while should_continue():
        try:
            data = audio_q.get(timeout=1.0)
            temp_buffer.append(data)
            now = time.time()

            if len(temp_buffer) >= 10:
                audio = np.concatenate(temp_buffer).flatten().astype(np.float32)
                result = model.transcribe(audio, language="en", fp16=False)
                text = result.get("text", "").strip()
                if text:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    line = f"[{timestamp}] {text}"
                    transcript_callback.emit(line)
                    buffer.append(line)
                    last_caption_time = now
                temp_buffer.clear()
            if is_auto_summary_enabled() and (
                now - last_caption_time >= PAUSE_SECONDS or
                now - last_summary_time >= get_summary_interval_sec()
            ):
                summary = summarize_buffer()
                print(f'summary: {summary}')
                if summary:
                    summary_callback.emit(summary)

        except queue.Empty:
            continue
def run_transcription(transcript_callback, summary_callback, should_continue, is_auto_summary_enabled, get_summary_interval_sec):
    global client
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("OpenAI API key is not set. Please go to Settings and add your key.")
    client = OpenAI(api_key=api_key)

    with sd.InputStream(channels=1, samplerate=samplerate,
                        callback=audio_callback, blocksize=blocksize):
        transcribe_stream(transcript_callback, summary_callback, should_continue, is_auto_summary_enabled, get_summary_interval_sec)

def load_api_key():
    return os.getenv("OPENAI_API_KEY", "")