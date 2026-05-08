import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import sys
from tqdm import tqdm

# Ensure Python can find the src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.sft_model import load_sft_model
from models.dummy_llm import DummyCausalLM
from models.dpo import get_batch_logps, simple_dpo_loss

def get_device():
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    return "cpu"

def get_logits(model, input_ids, use_sft):
    """Helper to handle the difference between HF models and our Dummy Model."""
    if use_sft:
        return model(input_ids).logits
    else:
        return model(input_ids)

def evaluate_accuracy(policy_model, test_loader, device, use_sft):
    """Calculates Golden Accuracy against the hidden truth in the Test Set."""
    policy_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set", leave=False):
            ids_A, ids_B = batch['input_ids_A'].to(device), batch['input_ids_B'].to(device)
            truth_is_A = batch['truth_is_A'].to(device)
            
            pi_A = get_batch_logps(get_logits(policy_model, ids_A, use_sft), ids_A)
            pi_B = get_batch_logps(get_logits(policy_model, ids_B, use_sft), ids_B)
            
            # If log-prob of A > B, the model predicts A (1). Else B (0).
            predictions = (pi_A > pi_B).int()
            correct += (predictions == truth_is_A).sum().item()
            total += truth_is_A.size(0)
            
    return correct / total

def train_baseline(epochs=15, batch_size=4, beta=0.1, use_sft=False):
    device = get_device()
    model_type = "GPT-2 (SFT)" if use_sft else "Dummy Transformer (Non-SFT)"
    print(f"Starting Baseline DPO Training on {device} using {model_type}...")

    # 1. Load Data
    train_dataset = torch.load("../../data/tokenized/train_tokens.pt")
    test_dataset = torch.load("../../data/tokenized/test_tokens.pt")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Load Models
    if use_sft:
        policy_model = load_sft_model("gpt2", device)
        ref_model = load_sft_model("gpt2", device)
    else:
        # Vocab size must be 50257 to match the GPT-2 tokenizer we used in data prep
        policy_model = DummyCausalLM(vocab_size=50257).to(device)
        ref_model = DummyCausalLM(vocab_size=50257).to(device)
        
    ref_model.eval() # Reference model is frozen forever
    optimizer = optim.AdamW(policy_model.parameters(), lr=1e-5)

    # 3. Setup Checkpointing
    os.makedirs("../../results", exist_ok=True)
    os.makedirs("../../models", exist_ok=True)
    metrics_history = []
    
    csv_filename = "../../results/baseline_sft_metrics.csv" if use_sft else "../../results/baseline_dummy_metrics.csv"
    model_filename = "../../models/baseline_gpt2.pth" if use_sft else "../../models/baseline_dummy.pth"

    # 4. Training Loop
    for epoch in range(epochs):
        policy_model.train()
        epoch_loss = 0.0
        
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        for batch in tqdm(train_loader, desc="Training"):
            ids_A, ids_B = batch['input_ids_A'].to(device), batch['input_ids_B'].to(device)
            votes = batch['vote'].to(torch.float32).to(device)
            
            optimizer.zero_grad()
            
            pi_A = get_batch_logps(get_logits(policy_model, ids_A, use_sft), ids_A)
            pi_B = get_batch_logps(get_logits(policy_model, ids_B, use_sft), ids_B)
            with torch.no_grad():
                ref_A = get_batch_logps(get_logits(ref_model, ids_A, use_sft), ids_A)
                ref_B = get_batch_logps(get_logits(ref_model, ids_B, use_sft), ids_B)
            
            # Simple DPO assumes A is the winner. If the vote was B (0), we invert the DPO arguments.
            loss = 0
            for i in range(len(votes)):
                if votes[i] == 1:
                    loss += simple_dpo_loss(pi_A[i:i+1], pi_B[i:i+1], ref_A[i:i+1], ref_B[i:i+1], beta)
                else:
                    loss += simple_dpo_loss(pi_B[i:i+1], pi_A[i:i+1], ref_B[i:i+1], ref_A[i:i+1], beta)
                    
            loss = loss / len(votes) # Mean batch loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        test_acc = evaluate_accuracy(policy_model, test_loader, device, use_sft)
        
        print(f"Train Loss: {avg_loss:.4f} | Test Golden Accuracy: {test_acc * 100:.2f}%")
        
        # Save metrics immediately
        metrics_history.append({"epoch": epoch + 1, "train_loss": avg_loss, "test_accuracy": test_acc})
        pd.DataFrame(metrics_history).to_csv(csv_filename, index=False)

    # 5. Save the Final Model Brain
    print("Training complete. Saving model...")
    torch.save(policy_model.state_dict(), model_filename)
    print(f"Baseline model saved to {model_filename}")

if __name__ == "__main__":
    # MAC TESTING: use_sft=False, epochs=2
    # KAGGLE TESTING: use_sft=True, epochs=15
    train_baseline(epochs=2, use_sft=True)