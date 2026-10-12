from transformers import AutoTokenizer, AutoModelForCausalLM

def expand_vocabulary(model_path="Qwen/Qwen-VL"):
    print(f"Loading tokenizer from {model_path}...")
    # NOTE: Qwen-VL uses tiktoken under the hood, but via transformers wrapper
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    special_tokens_dict = {'additional_special_tokens': ['[SEG]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    print(f"Added {num_added_toks} tokens. Vocaublary size is now {len(tokenizer)}")
    
    # Resize model embeddings to match new tokenizer size
    model.resize_token_embeddings(len(tokenizer))
    print("✅ Successfully expanded model embeddings with [SEG] token.")
    
    return tokenizer, model

if __name__ == "__main__":
    # Test script locally
    try:
        # In a real environment, uncomment to run:
        # tok, mod = expand_vocabulary()
        print("Tokenizer Expansion Module Ready.")
    except Exception as e:
        print(f"Skipping actual download during setup: {e}")
