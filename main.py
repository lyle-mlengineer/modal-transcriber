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
        audio_data: list[float]  # Audio data as a list of floats,
    ) -> str:

        transcription = self.model.transcribe(audio_data)["text"]

        return transcription
    
def read_audio_with_soundfile(filepath):
    data, samplerate = sf.read(filepath)
    return data, samplerate

@app.local_entrypoint()
def main(audio_path: str):
    import librosa

    audio_data, _ = librosa.load(audio_path, sr=16000)
    text = Transcribe().transcribe.remote(audio_data)
    print(f"Transcription: {text}")

    # To run the app in the cloud, use:

