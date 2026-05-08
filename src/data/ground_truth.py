import random
import pandas as pd
from datasets import load_dataset
import os

# Default sizes for paper-style runs (capped by HF split length in prepare_ground_truth).
DEFAULT_MAX_TRAIN = 10_000
DEFAULT_MAX_TEST = 2_000
DEFAULT_SEED = 42


def prepare_ground_truth(
    max_train=None,
    max_test=None,
    seed=DEFAULT_SEED,
    output_dir="../../data/processed/",
):
    """
    Load Anthropic/hh-rlhf, optionally subsample with a seeded shuffle from each official
    split (train rows only from 'train', test rows only from 'test'), then build CSVs
    with a random A/B assignment per row (reproducible given ``seed``).
    """
    print("Downloading Anthropic dataset (Train and Test splits)...")

    train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    test_dataset = load_dataset("Anthropic/hh-rlhf", split="test")

    if max_train is not None:
        n_train = min(int(max_train), len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=seed).select(range(n_train))
        print(f"Train: seeded shuffle (seed={seed}), kept {n_train} rows.")

    if max_test is not None:
        # Different stream seed so test order is not a simple offset of train shuffle
        test_shuffle_seed = seed + 1
        n_test = min(int(max_test), len(test_dataset))
        test_dataset = test_dataset.shuffle(seed=test_shuffle_seed).select(range(n_test))
        print(f"Test: seeded shuffle (seed={test_shuffle_seed}), kept {n_test} rows.")

    def process_split(dataset, split_name):
        raw_data = []
        for prompt_id, row in enumerate(dataset):
            truth_is_A = random.random() > 0.5
            raw_data.append(
                {
                    "prompt_id": prompt_id,
                    "text_A": row["chosen"] if truth_is_A else row["rejected"],
                    "text_B": row["rejected"] if truth_is_A else row["chosen"],
                    "truth_is_A": int(truth_is_A),
                }
            )
        return pd.DataFrame(raw_data)

    os.makedirs(output_dir, exist_ok=True)

    # Deterministic A/B swaps: consume RNG in train then test order
    random.seed(seed + 10_007)

    df_train = process_split(train_dataset, "train")
    df_train.to_csv(f"{output_dir}01_train_raw.csv", index=False)

    df_test = process_split(test_dataset, "test")
    df_test.to_csv(f"{output_dir}01_test_raw.csv", index=False)

    print(f"Saved {len(df_train)} prompts to 01_train_raw.csv")
    print(f"Saved {len(df_test)} prompts to 01_test_raw.csv")


if __name__ == "__main__":
    prepare_ground_truth(DEFAULT_MAX_TRAIN, DEFAULT_MAX_TEST, seed=DEFAULT_SEED)
