import torch
import pandas as pd
from transformers import AutoTokenizer
import os
from tqdm import tqdm

def run_oracle_tokenization():
   train_text = pd.read_csv("../../data/processed/01_train_raw.csv")
   tokenizer = AutoTokenizer.from_pretrained("gpt2")
   tokenizer.pad_token = tokenizer.eos_token
   os.makedirs("../../data/tokenized", exist_ok=True)
   oracle_tensors = []
   for _, row in tqdm(train_text.iterrows(), total=len(train_text), desc="Oracle Train Set"):
       oracle_tensors.append({
           "prompt_id": row['prompt_id'],
           "annotator_id": 999, # Dummy ID, Oracle doesn't have annotators
           "vote": float(row['truth_is_A']), # ORACLE REVEALS THE TRUTH
           "input_ids_A": tokenizer(row['text_A'], truncation=True, max_length=128, padding="max_length", return_tensors="pt")['input_ids'].squeeze(0),
           "input_ids_B": tokenizer(row['text_B'], truncation=True, max_length=128, padding="max_length", return_tensors="pt")['input_ids'].squeeze(0)
       })
   torch.save(oracle_tensors, "../../data/tokenized/oracle_train_tokens.pt")
   print("Saved oracle_train_tokens.pt")

if __name__ == "__main__":
   run_oracle_tokenization()