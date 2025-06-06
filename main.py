from fastapi import FastAPI, UploadFile, File
from models.stt_whisper import WhisperSTT
from utils.sentence_splitter import split_into_sentences

app = FastAPI()

stt_model = WhisperSTT("small")

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    text = stt_model.transcribe("temp.wav")
    sentences = split_into_sentences(text)
    results = []

    for sentence in sentences:
        print(sentence)

    return {
        "full_text": text,
        "results": results
    }
