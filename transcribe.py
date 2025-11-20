import modal
import requests
import librosa
import io
from pydantic import BaseModel
import whisper
import numpy as np
from oryks_google_drive import GoogleDrive


class TranscribeAudioRequest(BaseModel):
    file_id: str  


class TranscribeAudioResponse(BaseModel):
    transcription: str

volume = modal.Volume.from_name("ssflow-volume", create_if_missing=True)
mount_path = "/root/.cache/whisper"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("/home/lyle/.drive/credentials.json", "/root/.drive/credentials.json")
)

app = modal.App(name="whisper-transcribe-openai", image=image)


@app.cls(
    image=image,
    gpu="T4",
    volumes={mount_path: volume},
)
class SSFlowTranscriber:
    @modal.enter()
    def load_model(self):
        self.model = whisper.load_model("base")
        self.drive = GoogleDrive()
        self.drive.authenticate_from_credentials(
            credentials_path='/root/.drive/credentials.json'
        )

    def download_drive_audio(self, file_id: str) -> np.ndarray:
        file_content = self.drive.download_file_content(file_id)
        audio_data, _ = librosa.load(io.BytesIO(file_content), sr=16000)
        return audio_data
    
    
    @modal.fastapi_endpoint(
        method="POST",
        docs=True,
    )
    def transcribe(
        self,
        request: TranscribeAudioRequest,
    ) -> TranscribeAudioResponse:
        print(f"Received request to transcribe file ID: {request.file_id}")
        audio = self.download_drive_audio(request.file_id)
        print(f"Downloaded audio data with length: {len(audio)} samples")
        transcription = self.model.transcribe(audio)["text"]
        print(f"Transcription completed: {transcription}")
        return TranscribeAudioResponse(transcription=transcription)
    

@app.local_entrypoint()
def main():
    import requests

    transcriber = SSFlowTranscriber()
    url = transcriber.transcribe.get_web_url()

    audio_url = "https://github.com/lyle-mlengineer/modal-transcriber/raw/refs/heads/main/maongezi.wav"
    response = requests.post(url, json={"audio_url": audio_url})
    print(response.json())