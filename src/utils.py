import base64
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os

# Initialize clients
llm_vision = ChatOpenAI(model="gpt-4o", max_tokens=1000)
client = OpenAI()


def encode_image(image_file):
    """Converts uploaded file to base64 for the API."""
    return base64.b64encode(image_file.read()).decode('utf-8')


def perform_ocr(image_file):
    """Extracts text from image using GPT-4o Vision."""
    base64_image = encode_image(image_file)
    response = llm_vision.invoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this math problem exactly into LaTeX/text. Do NOT solve it."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ]
    )
    return response.content


def transcribe_audio(audio_file):
    """Transcribes audio using Whisper."""
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return transcription.text
