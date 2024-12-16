# How to run ai_communicator.py

Because the latest coqui-ai/TTS can run with python 3.10.8 (now 2024/12/10), you should switch the python to this version. If you don't know the right python version, you can docker pull the TTS image and check the python version in this image.

I don't buy the official ChatGPT plus. So I run [chat2api](https://github.com/lanqian528/chat2api) to proxy the chat API to the chatgpt web.

```bash
git clone https://github.com/lanqian528/chat2api.github
cd chat2api
RETRY_TIMES=1 AUTHORIZATION='abc'  python app.py
```

```bash
pyenv install 3.10.8
pyenv vivirtualenv 3.10.8 tts-ai
pyenv activate tts-ai

# install the required packages
pip install TTS
pip install pyaudio pynput 
pip install pyttsx3
pip install git+https://github.com/openai/whisper.git

# prepare a wav that is the voice of your favorite speaker and save it to /tmp/example.mp3.
ffmpeg -i source_audio.mp3 -ss 00:08:36 -to 00:08:55 -acodec copy /tmp/example.mp3

python ai_communicator.py
```
