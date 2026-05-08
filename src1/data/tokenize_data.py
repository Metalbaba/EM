import torch
import pandas as pd
from transformers import AutoTokenizer
import os
from tqdm import tqdm

def tokenize_csv(df_text, df_votes, tokenizer, max_length=128):
    processed_data = []
    
    if df_votes is None:
        for idx, row in tqdm(df_text.iterrows(), total=len(df_text), desc="Tokenizing Test Set"):
            tokens_A = tokenizer(row['text_A'], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            tokens_B = tokenizer(row['text_B'], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            
            processed_data.append({
                "prompt_id": row['prompt_id'],
                "input_ids_A": tokens_A['input_ids'].squeeze(0),
                "input_ids_B": tokens_B['input_ids'].squeeze(0),
                "truth_is_A": row['truth_is_A'] 
            })
    else:
        for idx, vote_row in tqdm(df_votes.iterrows(), total=len(df_votes), desc="Tokenizing Train Set"):
            prompt_id = int(vote_row['prompt_id'])
            text_row = df_text[df_text['prompt_id'] == prompt_id].iloc[0]
            
            tokens_A = tokenizer(text_row['text_A'], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            tokens_B = tokenizer(text_row['text_B'], truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")

            processed_data.append({
                "prompt_id": prompt_id,
                "annotator_id": int(vote_row['annotator_id']),
                "vote": float(vote_row['vote']),
                "input_ids_A": tokens_A['input_ids'].squeeze(0),
                "input_ids_B": tokens_B['input_ids'].squeeze(0)
            })
            
    return processed_data

def run_tokenization():
    print("Loading CSVs...")
    train_text = pd.read_csv("../../data/processed/01_train_raw.csv")
    test_text = pd.read_csv("../../data/processed/01_test_raw.csv")
    train_votes = pd.read_csv("../../data/processed/02_train_noisy_votes.csv")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    os.makedirs("../../data/tokenized", exist_ok=True)
    
    train_tensors = tokenize_csv(train_text, train_votes, tokenizer)
    torch.save(train_tensors, "../../data/tokenized/noisy_train_tokens.pt")

    test_tensors = tokenize_csv(test_text, None, tokenizer)
    torch.save(test_tensors, "../../data/tokenized/test_tokens.pt")

    print("Saved noisy_train_tokens.pt and test_tokens.pt successfully!")

if __name__ == "__main__":
    run_tokenization()