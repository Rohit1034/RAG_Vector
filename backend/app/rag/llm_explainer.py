import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def generate_explanation(prediction, severity, context, patient_data):
    prompt = f"""
You are a medical assistant.

Patient data:
{patient_data}

Model prediction: {severity}

Relevant medical knowledge:
{context}

Explain in simple terms:
- Why this prediction happened
- Which factors contributed
- What it means for the patient

Keep it short and clear.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content