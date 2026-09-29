from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    bad = {label: tokenizer.encode(label, add_special_tokens=False) for label in "ABCDEFGHIJKLMNOPQRST" if len(tokenizer.encode(label, add_special_tokens=False)) != 1}
    if bad:
        raise RuntimeError(f"answer labels must be single tokens: {bad}")
    print("OK", {label: tokenizer.encode(label, add_special_tokens=False)[0] for label in "ABCDEFGHIJKLMNOPQRST"})

if __name__ == "__main__": main()
