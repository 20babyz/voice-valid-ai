from models.phishing_detect import PhishingDetector
from fastapi import FastAPI, UploadFile, File
from models.stt_whisper import WhisperSTT
from utils.sentence_splitter import split_into_sentences

app = FastAPI()

stt_model = WhisperSTT("small")
phishing_model = PhishingDetector("models/phishing_kobert_state.pt")

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    text = stt_model.transcribe("temp.wav")
    sentences = split_into_sentences(text)
    results = []

    for sentence in sentences:
        score = phishing_model.predict(sentence)
        results.append({
            "sentence": sentence,
            "score": round(score, 4),
            "is_phishing": score > 0.5
        })

    return {
        "full_text": text,
        "results": results
    }
