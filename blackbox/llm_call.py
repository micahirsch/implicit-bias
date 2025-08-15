import google.generativeai as genai
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


def call_model(
    prompt: list[str],
    model_name: str = "gemini-1.5-flash-001",
    temp: float = 0.0,
):
    if "gemini" in model_name:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        gemini_model = genai.GenerativeModel(model_name=model_name)
        generation_config = genai.GenerationConfig(temperature=temp)
        full_response = gemini_model.generate_content(
            contents=prompt[0]["content"], 
            generation_config=generation_config
        )
        return full_response.text
    elif "gpt" in model_name or "o4-mini" in model_name:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if "o4-mini" in model_name:
            completion = client.chat.completions.create(
                model=model_name,
                messages=prompt,
            )
        else:
            completion = client.chat.completions.create(
                model=model_name,
                messages=prompt,
                temperature=temp,
            )

        return completion.choices[0].message.content
    else:
        raise ValueError(f"Model {model_name} not supported.")