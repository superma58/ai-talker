
import audioop
import whisper
import pyaudio
import numpy as np
import io
import queue
import threading
import time
import wave

model = whisper.load_model("tiny.en")

p = pyaudio.PyAudio()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 16000

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


frames = []
audio_queue = queue.Queue()


SILENCE_THRESHOLD = 200  # 音量阈值，低于此值认为是停顿
SILENCE_DURATION = 0.7  # 停顿持续时间（秒）


def detect_silence(audio_data):
    """检测音频数据中的静音部分"""
    rms = audioop.rms(audio_data, 2)  # 计算音频的RMS（根均方）值
    # print("RMS: ", rms)
    return rms < SILENCE_THRESHOLD


def record_audio():
    """录制音频并将音频数据传递到队列"""
    print("开始实时录音并识别语音...")
    frames = []
    last_audio_time = time.time()

    while True:
        data = stream.read(CHUNK)
        if detect_silence(data):
            if time.time() - last_audio_time > SILENCE_DURATION and frames:
                audio_queue.put(frames)
                frames = []  # 重置缓冲区
        else:
            frames.append(data)
            last_audio_time = time.time()

    # while True:
    #     frames = []
    #     for i in range(2):
    #         data = stream.read(CHUNK)
    #         frames.append(data)
    #     audio_queue.put(frames)


def recognize_and_translate():
    """从队列中获取音频数据，识别并翻译输出"""
    while True:
        if not audio_queue.empty():
            # 获取音频数据
            frames = audio_queue.get()
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).flatten().astype(np.float32) / 32768.0
            result = model.transcribe(audio=audio_data, language="en")

            # audio_bytes = io.BytesIO(data)  # 使用 BytesIO 处理数据
            # audio = whisper.load_audio(audio_bytes)
            # audio = whisper.pad_or_trim(audio)
            # result = model.transcribe(audio)
            print(f"识别结果: {result['text']}")
        time.sleep(0.1)


record_thread = threading.Thread(target=record_audio)
recognize_thread = threading.Thread(target=recognize_and_translate)

record_thread.start()
recognize_thread.start()

# 保持主线程活跃
record_thread.join()
recognize_thread.join()
