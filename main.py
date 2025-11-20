import modal
import soundfile as sf


image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_requirements("requirements.txt")
)

app = modal.App(name="whisper-transcribe-openai", image=image)

@app.cls(
    image=image,
    gpu="T4",
)
class Transcribe:
    @modal.enter()
    def load_model(self):
        import whisper

        self.model = whisper.load_model("base")

    @modal.method()
    def transcribe(
        self,
        audio_url: str, # Audio data as a list of floats,
    ) -> str:
        audio, samplerate = self.download_audio(audio_url)
        transcription = self.model.transcribe(audio)["text"]

        return transcription
    
    def download_audio(self, url: str) -> bytes:
        import requests
        import librosa
        import io

        response = requests.get(url)
        response.raise_for_status()  # Ensure we raise an error for bad responses
        audio_data, samplerate = librosa.load(io.BytesIO(response.content), sr=None)
        return audio_data, samplerate
    

@app.local_entrypoint()
def main():

    audio_url: str = "https://github.com/lyle-mlengineer/modal-transcriber/raw/refs/heads/main/maongezi.wav"
    text = Transcribe().transcribe.local(audio_url)
    print(f"Transcription: {text}")

    # To run the app in the cloud, use:

