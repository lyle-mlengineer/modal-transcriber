import requests
import librosa
import soundfile as sf
import io
from oryks_google_drive import GoogleDrive
import numpy as np

# def download_audio(url: str) -> bytes:
#     response = requests.get(url)
#     response.raise_for_status()  # Ensure we raise an error for bad responses
#     return librosa.load(io.BytesIO(response.content), sr=None)

# if __name__ == "__main__":
#     audio_url = "http://192.168.1.228:8000/audio_input/gwrxOHg6A20_1.wav"
#     audio_data, samplerate = download_audio(audio_url)
#     print(f"Downloaded audio with sample rate: {samplerate} Hz")
#     print(f"Audio data length: {len(audio_data)} samples")
client_secrets_file = '/home/lyle/Downloads/secret.json'
drive = GoogleDrive()
# drive.authenticate(client_secret_file=client_secrets_file)
drive.authenticate_from_credentials(
    credentials_path='/home/lyle/.drive/credentials.json'
)

def download_audio_from_gdrive(file_id: str) -> tuple[np.ndarray, int]:
    file_content = drive.download_file_content(file_id)
    audio_data, samplerate = librosa.load(io.BytesIO(file_content), sr=16000)
    return audio_data, samplerate

file_id = '1alI9X1YZRa2XY6eBMOMXWbOYXTiEPrxg'
audio_data, samplerate = download_audio_from_gdrive(file_id)
print(f"Downloaded audio from Google Drive with sample rate: {samplerate} Hz")
print(f"Audio data length: {len(audio_data)} samples")
# sf.write('downloaded_audio.wav', audio_data, samplerate)