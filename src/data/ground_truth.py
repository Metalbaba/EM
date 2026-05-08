import random
import pandas as pd
from datasets import load_dataset
import os

def prepare_ground_truth(max_train=None, max_test=None, seed = 42, output_dir="../../data/processed/"):
    print(f"Downloading Anthropic dataset (Train and Test splits)...| Seed = {seed}")
    random.seed(seed)

    train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    test_dataset = load_dataset("Anthropic/hh-rlhf", split="test")
    
    if max_train: 
        train_dataset = train_dataset.shuffle(seed=seed).select(range(min(max_train, len(train_dataset))))
    if max_test: 
        test_dataset = test_dataset.shuffle(seed=seed).select(range(min(max_test, len(test_dataset))))
        
    def process_split(dataset, split_name):
        raw_data = []
        for prompt_id, row in enumerate(dataset):
            truth_is_A = random.random() > 0.5
            raw_data.append({
                "prompt_id": prompt_id,
                "text_A": row['chosen'] if truth_is_A else row['rejected'],
                "text_B": row['rejected'] if truth_is_A else row['chosen'],
                "truth_is_A": int(truth_is_A)
            })
        return pd.DataFrame(raw_data)

    os.makedirs(output_dir, exist_ok=True)
    
    df_train = process_split(train_dataset, "train")
    df_train.to_csv(f"{output_dir}01_train_raw.csv", index=False)
    
    df_test = process_split(test_dataset, "test")
    df_test.to_csv(f"{output_dir}01_test_raw.csv", index=False)
    
    print(f"Saved {len(df_train)} prompts to 01_train_raw.csv")
    print(f"Saved {len(df_test)} prompts to 01_test_raw.csv")

if __name__ == "__main__":
    prepare_ground_truth(10000, 2000, 42)