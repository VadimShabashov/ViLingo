# ViLingo

# Requirements

## Python 3.10

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10
```

## Libraries

```bash
python3.10 -m pip install git+https://github.com/m-bain/whisperx.git
python3.10 -m pip install transformers==4.33.0
python3.10 -m pip install demucs==4.0.1
python3.10 -m pip install TTS==0.20.3
python3.10 -m pip install pydub==0.25.1
sudo apt install ffmpeg
```

Order is important for avoiding dependency errors.

# Hardware requirements

The code was tested on Tesla-V100 1x32GB.
