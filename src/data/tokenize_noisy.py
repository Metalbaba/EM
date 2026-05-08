import torch
import pandas as pd
from transformers import AutoTokenizer
import os
from tqdm import tqdm

def run_tokenization():
    train_text = pd.read_csv("../../data/processed/01_train_raw.csv")
    test_text = pd.read_csv("../../data/processed/01_test_raw.csv")
    train_votes = pd.read_csv("../../data/processed/02_train_noisy_votes.csv")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    os.makedirs("../../data/tokenized", exist_ok=True)
    
    # 1. Tokenize Test Set (Only needs text and truth)
    test_tensors = []
    for _, row in tqdm(test_text.iterrows(), total=len(test_text), desc="Test Set"):
        test_tensors.append({
            "prompt_id": row['prompt_id'],
            "input_ids_A": tokenizer(row['text_A'], truncation=True, max_length=128, padding="max_length", return_tensors="pt")['input_ids'].squeeze(0),
            "input_ids_B": tokenizer(row['text_B'], truncation=True, max_length=128, padding="max_length", return_tensors="pt")['input_ids'].squeeze(0),
            "truth_is_A": row['truth_is_A'] 
        })
    torch.save(test_tensors, "../../data/tokenized/test_tokens.pt")

    # 2. Tokenize Noisy Train Set
    train_tensors = []
    for _, vote_row in tqdm(train_votes.iterrows(), total=len(train_votes), desc="Noisy Train Set"):
        prompt_id = int(vote_row['prompt_id'])
        text_row = train_text[train_text['prompt_id'] == prompt_id].iloc[0]
        train_tensors.append({
            "prompt_id": prompt_id,
            "annotator_id": int(vote_row['annotator_id']),
            "vote": float(vote_row['vote']),
            "input_ids_A": tokenizer(text_row['text_A'], truncation=True, max_length=128, padding="max_length", return_tensors="pt")['input_ids'].squeeze(0),
            "input_ids_B": tokenizer(text_row['text_B'], truncation=True, max_length=128, padding="max_length", return_tensors="pt")['input_ids'].squeeze(0)
        })
    torch.save(train_tensors, "../../data/tokenized/noisy_train_tokens.pt")
    print("Saved test_tokens.pt and noisy_train_tokens.pt")

if __name__ == "__main__":
    run_tokenization()