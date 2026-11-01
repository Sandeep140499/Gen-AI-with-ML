
import warnings
from transformers import AutoTokenizer

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("       GPT-2 Tokenization Demo")
print("=" * 50)

# Load the pretrained GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Get user input
text = input("\nEnter a sentence: ").strip()

# Check for empty input
if not text:
    print("\nError: Please enter a valid sentence.")
    exit()

# Generate tokens and token IDs
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

# Display results
print("\n" + "=" * 50)
print("Original Text")
print("=" * 50)
print(text)

print("\nTokens")
print("=" * 50)
print(tokens)

print("\nToken IDs")
print("=" * 50)
print(token_ids)

print("\nTotal Number of Tokens")
print("=" * 50)
print(len(token_ids))
print("=" * 50)
