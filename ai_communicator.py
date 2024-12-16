import json
import os
import requests
import time

import numpy as np
import pyttsx3
import pyaudio
import warnings
import whisper
from pynput import keyboard
from threading import Thread

is_recording = False
is_running = True

api_key = "abc"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
url = "http://localhost:5005/v1/chat/completions"
conversation_id = ""
message_id = ""

model = whisper.load_model("tiny.en")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper requires 16kHz
CHUNK = 1024

# Init PyAudio
audio = pyaudio.PyAudio()

# Init 
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 200)
tts_engine.setProperty('voice', 'com.apple.voice.compact.en-US.Samantha')

warnings.filterwarnings("ignore", category=UserWarning)


# Init coqui-ai TTS
import torch
from TTS.api import TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts_ai = None
# for i in TTS().list_models().list_models(): print(i)
# tts_ai = TTS("tts_models/en/ljspeech/speedy-speech").to(device)
# tts_ai = TTS("tts_models/en/ljspeech/glow-tts").to(device)
tts_ai = TTS("tts_models/en/ljspeech/fast_pitch").to(device)


def stop_audio():
    global audio
    if audio:
        audio.terminate()


def say_something(msg):
    if tts_ai:
        wav = tts_ai.tts(text=msg, speaker_wav="/tmp/example.mp3")
        if torch.is_tensor(wav):
            wav = wav.cpu().numpy()
        if isinstance(wav, list):
            wav = np.array(wav)
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        wav_norm = wav_norm.astype(np.int16)
        s = audio.open(format=pyaudio.paInt16, channels=1, rate=tts_ai.synthesizer.output_sample_rate, output=True)
        s.write(wav_norm.tobytes())
        s.stop_stream()
        s.close()
        return
    if tts_engine._inLoop:
        tts_engine.endLoop()
    tts_engine.say(msg)
    tts_engine.runAndWait()


def send_chat(message, conversation_id="", message_id=""):
    data = {
        "messages":
        [
            {
                "role": "user",
                "content": message
            }
        ],
        "stream": True,
        "model": "gpt-4o-mini",
        "history_disabled": False,
        "temperature": 0.5,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "top_p": 1,
        "max_tokens": 4000
    }
    if conversation_id:
        data["conversation_id"] = conversation_id
    if message_id:
        data["message_id"] = message_id
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response


def parse_chat_resp(response, conversation_id, message_id):
    if response.status_code != 200:
        print("Response error:", response.status_code)
        return "", conversation_id, message_id
    conversation_id = ""
    message_id = ""
    message = ""
    for chunk in response.text.split("\n"):
        if not chunk.startswith("data: ") or chunk == "data: [DONE]":
            continue
        chunk_data = json.loads(chunk[6:])
        message_id = chunk_data.get("message_id", "")
        conversation_id = chunk_data.get("conversation_id", "")
        if chunk_data.get("choices"):
            message += chunk_data["choices"][0]["delta"].get("content", "")
    # print(conversation_id, message_id, message)
    return message, conversation_id, message_id


def chat(message, speak=True):
    global conversation_id, message_id
    # chat with AI
    print("current conversation:", conversation_id, message_id)
    response = send_chat(message, conversation_id, message_id)
    if not response:
        print("Fail to chat")
        return False
    ai_content, conversation_id, message_id = parse_chat_resp(response, conversation_id, message_id)
    print("new conversation", conversation_id, message_id)
    with open('chat_token.json', 'w') as f:
        json.dump({'conversation_id': conversation_id, 'message_id': message_id}, f)
    if ai_content:
        print("Assistant: ", ai_content)
        if speak:
            say_something(ai_content)
        return True
    else:
        return False


def _interact():
    global is_recording, CHUNK, model
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    while is_recording:
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    # in case, user press esc when recording
    if not is_running:
        return
    # translate the audio
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).flatten().astype(np.float32) / 32768.0
    result = model.transcribe(audio=audio_data, language="en")
    if result:
        text = result.get('text')
        print("You said: ", text)
    else:
        print("No found words")
        return
    chat(text)


def interact():
    while is_running:
        if not is_recording:
            time.sleep(0.1)
            continue
        _interact()


def on_press(key):
    global is_recording
    try:
        if key == keyboard.Key.shift:
            if tts_engine._inLoop:
                tts_engine.endLoop()
            if not is_recording:
                is_recording = True
                 # print("Press key to notify the record...")
    except AttributeError:
        pass


def on_release(key):
    global is_recording, is_running
    if key == keyboard.Key.shift:
        is_recording = False
        # print("Release key to stop recording.")
    if key == keyboard.Key.esc:
        is_running = False
        is_recording = False
        stop_audio()
        return False


ready = False
if os.path.exists('chat_token.json'):
    with open('chat_token.json', 'r') as f:
        data = json.load(f)
        conversation_id, message_id = data['conversation_id'], data['message_id']
    # ready = chat("Let's start a new conversation. I'll start later.", speak=False)
    ready = True
if not ready:
    conversation_id = message_id = ""
    chat("I'm Nick. You are a professional English teacher. I'll talk with you. You can reply me, check my English fault, find some topics to discuss with me. My input may have some wrong words, especially the complex words. These wrong words are caused by my pronounce. You can ignore them.", False)


time_thread = Thread(target=interact, daemon=True)
time_thread.start()

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

