import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load environment variables from the project folder
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# Read the API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        f"GEMINI_API_KEY not found. Check your .env file at: {env_path}"
    )

# Create Gemini client
client = genai.Client(api_key=api_key)

# Preferred models (first available one will be used)
PREFERRED_MODELS = [
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

print("=" * 50)
print("       Text Generation with Gemini")
print("=" * 50)

# Find an available model
available_models = {
    model.name.replace("models/","")
    for model in client.models.list()
}

selected_model = None

for model in PREFERRED_MODELS:
    if model in available_models:
        selected_model = model
        break

if not selected_model:
    print("\nNo supported Gemini model was found.")
    print("\nAvailable models:")
    for model in sorted(available_models):
        print(f"- {model}")
    exit()

print(f"\nUsing Model: {selected_model}")

while True:
    prompt = input("\nEnter your prompt (type 'exit' to quit): ")

    if prompt.lower() == "exit":
        print("Thank you for using Gemini!")
        break

    try:
        response = client.models.generate_content(
            model=selected_model,
            contents=prompt,
        )
        print(f"\nGemini: {response.text}")

    except Exception as e:
        print(f"\nError: {e}")