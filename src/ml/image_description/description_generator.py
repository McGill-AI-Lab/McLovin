import google.generativeai as genai
from dotenv import load_dotenv
import os

def configure_generative_ai(api_key_env: str = "celeba_key"):
    """Load the key from environment and configure genai."""
    load_dotenv()
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"API key not found in environment variable: {api_key_env}")
    genai.configure(api_key=api_key)

def generate_description(traits):
    """
    Use the Google Generative AI to produce a textual description
    based on a list of traits.
    """
    instructions = (
        "Write 1-3 concise sentences on the user's portrait with "
        f"the following physical attributes: {traits}"
    )

    # Example usage with generative AI
    model = genai.GenerativeModel(
        "gemini-1.5-flash", system_instruction=instructions
    )

    response = model.generate_content(
        "Write the sentences",
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=100,
            temperature=1.0,
        ),
    )

    return response.text
