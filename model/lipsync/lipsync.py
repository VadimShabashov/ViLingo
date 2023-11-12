import subprocess
# from base64 import b64decode
# import numpy as np
# from scipy.io.wavfile import read as wav_read
# import ffmpeg
# import io


class LipSyncModel:
    def __init__(self):
        self.init_commands = "git clone https://github.com/inspired99/Wav2Lip; wget 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA' -O './Wav2Lip/checkpoints/wav2lip_gan.pth'; pip install https://raw.githubusercontent.com/AwaleSajil/ghc/master/ghc-1.0-py3-none-any.whl; cd ./Wav2Lip; pip install -r requirements.txt; wget 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth' -O  ./Wav2Lip/face_detection/detection/sfd/s3fd.pth';"

    def __call__(self, path_to_video, path_to_audio):
        result_1 = subprocess.run(self.init_commands, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # !pip install - q youtube - dl
        # !pip install ffmpeg - python
        # !pip install librosa == 0.9.1

        result_2 = subprocess.run(
            f"cd ./Wav2Lip; python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face '{path_to_video}' --audio '{path_to_audio}'",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)