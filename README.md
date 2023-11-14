# ViLingo

The repository contains a pipeline for translating russian video into a subset of the most popular
languages, such as english, french, italian, portuguese...

The following models were used:
* Demucs - for separating vocal from other sounds
* Whisper - for performing STT and getting timestamps for each phrase
* NLLB-200 - for translation of the text
* Coqui xtts_v2 - for TTS and voice cloning

Note, that each phrase is processed separately, which helps to make pronounce each phrase with the voice of 
the corresponding speaker.

Additionally, we tried to implement and test lipsync, but it wasn't fully integrated into our pipeline.
The reason for this is that the lipsync model doesn't perform well in case of multiple people on a video.
The model implementation you can find in the branch `lipsync`.


# Examples

Please find some examples of our work
[there](https://drive.google.com/drive/folders/1LqOT3hCsz6AI9shP1lP4ya5DxC1VzaW-?usp=drive_link).

# Running

Run the following command from `model` directory:
```bash
python3.10 main.py [path_to_video] [language]
```

Language can be `en` or `fr`, for example.

# Requirements

## System

The code was tested on Ubuntu 22.04.01 .

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

Several models, that are used in our project, are having dependency conflicts. As a workaround we suggest installing
them sequentially without requirements.txt (that worked for us).

# Hardware requirements

RAM: 16GB
GPU: Tesla-V100 1x32GB.
