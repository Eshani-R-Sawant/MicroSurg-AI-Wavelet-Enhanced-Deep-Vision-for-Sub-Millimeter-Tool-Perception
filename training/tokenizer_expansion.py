"""
Tokenizer Vocabulary Expansion for Special Surgical Tokens.

Adds [SEG] special token to the Qwen-VL tokenizer and resizes model embeddings.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM


def expand_vocabulary(model_path: str = "Qwen/Qwen-VL", new_tokens: list = None):
    if new_tokens is None:
        new_tokens = ["[SEG]"]

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    special_tokens_dict = {"additional_special_tokens": new_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    print(f"Added {num_added_toks} tokens. Vocabulary size is now {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))
    print(f"✅ Successfully expanded model embeddings with {new_tokens}.")
    return tokenizer, model


if __name__ == "__main__":
    print("Tokenizer Expansion Module Ready.")
