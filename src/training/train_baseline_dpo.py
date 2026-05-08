import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.sft_model import load_sft_model
from models.dummy_llm import DummyCausalLM
from models.dpo import get_batch_logps, simple_dpo_loss

def get_device():
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    return "cpu"

def get_logits(model, input_ids, use_sft):
    return model(input_ids).logits if use_sft else model(input_ids)

def evaluate_accuracy(policy_model, test_loader, device, use_sft):
    policy_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set", leave=False):
            ids_A, ids_B = batch['input_ids_A'].to(device), batch['input_ids_B'].to(device)
            truth_is_A = batch['truth_is_A'].to(device)
            pi_A = get_batch_logps(get_logits(policy_model, ids_A, use_sft), ids_A)
            pi_B = get_batch_logps(get_logits(policy_model, ids_B, use_sft), ids_B)
            predictions = (pi_A > pi_B).int()
            correct += (predictions == truth_is_A).sum().item()
            total += truth_is_A.size(0)
    return correct / total

def train_baseline(epochs=15, batch_size=4, beta=0.1, use_sft=False, is_oracle=False):
    device = get_device()
    base_name = "gpt2" if use_sft else "dummy"
    prefix = "oracle" if is_oracle else "baseline"
    
    print(f"Starting {prefix.upper()} DPO Training on {device} using {base_name}...")

    # 1. Dynamic Data Loading (Oracle vs Noisy)
    dataset_name = "oracle_train_tokens.pt" if is_oracle else "noisy_train_tokens.pt"
    train_dataset = torch.load(f"../../data/tokenized/{dataset_name}")
    test_dataset = torch.load("../../data/tokenized/test_tokens.pt")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Load Models
    if use_sft:
        policy_model = load_sft_model("gpt2", device)
        ref_model = load_sft_model("gpt2", device)
    else:
        policy_model = DummyCausalLM(vocab_size=50257).to(device)
        ref_model = DummyCausalLM(vocab_size=50257).to(device)
        
    ref_model.eval()
    optimizer = optim.AdamW(policy_model.parameters(), lr=1e-5)

    # 3. Dynamic Checkpointing Paths
    os.makedirs("../../results", exist_ok=True)
    os.makedirs("../../models", exist_ok=True)
    
    csv_filename = f"../../results/{prefix}_{base_name}_metrics.csv"
    model_filename = f"../../models/{prefix}_{base_name}.pth"
    metrics_history = []

    # 4. Training Loop
    for epoch in range(epochs):
        policy_model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            ids_A, ids_B = batch['input_ids_A'].to(device), batch['input_ids_B'].to(device)
            votes = batch['vote'].to(torch.float32).to(device)
            
            optimizer.zero_grad()
            pi_A = get_batch_logps(get_logits(policy_model, ids_A, use_sft), ids_A)
            pi_B = get_batch_logps(get_logits(policy_model, ids_B, use_sft), ids_B)
            with torch.no_grad():
                ref_A = get_batch_logps(get_logits(ref_model, ids_A, use_sft), ids_A)
                ref_B = get_batch_logps(get_logits(ref_model, ids_B, use_sft), ids_B)
            
            loss = 0
            for i in range(len(votes)):
                if votes[i] == 1:
                    loss += simple_dpo_loss(pi_A[i:i+1], pi_B[i:i+1], ref_A[i:i+1], ref_B[i:i+1], beta)
                else:
                    loss += simple_dpo_loss(pi_B[i:i+1], pi_A[i:i+1], ref_B[i:i+1], ref_A[i:i+1], beta)
                    
            loss = loss / len(votes)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        test_acc = evaluate_accuracy(policy_model, test_loader, device, use_sft)
        print(f"Train Loss: {avg_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")
        
        metrics_history.append({"epoch": epoch + 1, "train_loss": avg_loss, "test_accuracy": test_acc})
        pd.DataFrame(metrics_history).to_csv(csv_filename, index=False)

    torch.save(policy_model.state_dict(), model_filename)
    print(f"Saved: {model_filename} and {csv_filename}")

if __name__ == "__main__":
    # Test combinations
    train_baseline(epochs=1, use_sft=False, is_oracle=False) # Noisy Baseline Dummy
    # train_baseline(epochs=1, use_sft=False, is_oracle=True)  # Oracle Dummy