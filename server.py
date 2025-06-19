#!/usr/bin/env python3
"""
ocr_genre_server.py
POST /ocr_genre { "image": "<base64-jpeg>" }
→ {"text": "...", "genre": "...", "scores": {...}}
"""
import base64, io, numpy as np, torch
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from paddleocr import PaddleOCR
from transformers import pipeline
import uvicorn

GENRE_LABELS = [
    "sports", "news", "cooking", "travel",
    "music", "drama", "comedy", "documentary",
]

# -------- 모델 로드 (1회) ---------------------------------------------------
ocr_engine = PaddleOCR(use_angle_cls=False, lang="en")
device = 0 if torch.cuda.is_available() else -1
genre_clf = pipeline("zero-shot-classification",
                     model="facebook/bart-large-mnli",
                     device=device)

# -------- FastAPI -----------------------------------------------------------
app = FastAPI(title="OCR + Genre Service")

class Req(BaseModel):
    image: str                # base64-encoded JPEG

@app.post("/ocr_genre")
def ocr_genre(req: Req):
    img = Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB")
    arr = np.array(img)

    # OCR
    best_txt, best_score = "", 0.0
    for res in (ocr_engine.ocr(arr) or []):
        for txt, sc in zip(res["rec_texts"], res["rec_scores"]):
            if sc > best_score: best_txt, best_score = txt, sc

    # Genre
    prompt = f"Text on screen: {best_txt}"
    res    = genre_clf(prompt, GENRE_LABELS, multi_label=False)
    genre  = res["labels"][0]

    return {"text": best_txt,
            "text_score": best_score,
            "genre": genre,
            "scores": dict(zip(res["labels"], map(float, res["scores"])))}

# -------- main --------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)