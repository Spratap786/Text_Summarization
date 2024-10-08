# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model_loader import generate_summary

app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Text Summarization API!"}

class TextRequest(BaseModel):
    text: str

@app.post("/summarize/")
def summarize(request: TextRequest):
    try:
        summary = generate_summary(request.text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
