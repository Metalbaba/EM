import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.sft_model import load_sft_model
from models.dummy_llm import DummyCausalLM
from models.dpo import get_batch_logps, projected_dpo_loss
from models.em_standalone import DawidSkeneEM # Import your EM class!

def get_device():
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    return "cpu"

def get_logits(model, input_ids, use_sft):
    return model(input_ids).logits if use_sft else model(input_ids)

def get_llm_priors(policy_model, ref_model, dataloader, num_prompts, device, use_sft, beta=0.1):
    """Generates the LLM's confidence scores to feed back into the EM algorithm."""
    policy_model.eval()
    priors = np.full(num_prompts, 0.5)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating LLM Priors", leave=False):
            ids_A, ids_B = batch['input_ids_A'].to(device), batch['input_ids_B'].to(device)
            p_ids = batch['prompt_id'].numpy()
            
            pi_A = get_batch_logps(get_logits(policy_model, ids_A, use_sft), ids_A)
            pi_B = get_batch_logps(get_logits(policy_model, ids_B, use_sft), ids_B)
            ref_A = get_batch_logps(get_logits(ref_model, ids_A, use_sft), ids_A)
            ref_B = get_batch_logps(get_logits(ref_model, ids_B, use_sft), ids_B)
            
            logits = (pi_A - pi_B) - (ref_A - ref_B)
            prob_A = torch.sigmoid(beta * logits).cpu().numpy()
            
            for i, p_id in enumerate(p_ids):
                priors[p_id] = prob_A[i]
                
    return priors

def evaluate_accuracy(policy_model, test_loader, device, use_sft):
    policy_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            ids_A, ids_B = batch['input_ids_A'].to(device), batch['input_ids_B'].to(device)
            truth_is_A = batch['truth_is_A'].to(device)
            
            pi_A = get_batch_logps(get_logits(policy_model, ids_A, use_sft), ids_A)
            pi_B = get_batch_logps(get_logits(policy_model, ids_B, use_sft), ids_B)
            predictions = (pi_A > pi_B).int()
            correct += (predictions == truth_is_A).sum().item()
            total += truth_is_A.size(0)
    return correct / total

def train_projected_dpo(epochs=15, batch_size=4, beta=0.1, use_sft=False, use_joint_em=True):
    device = get_device()
    print(f"Starting Projected DPO | Joint Mode: {use_joint_em} | Device: {device}")

    # 1. Load Data
    train_dataset = torch.load("../../data/tokenized/train_tokens.pt")
    test_dataset = torch.load("../../data/tokenized/test_tokens.pt")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    df_votes = pd.read_csv("../../data/processed/02_train_noisy_votes.csv")
    num_prompts = df_votes['prompt_id'].max() + 1
    num_annotators = df_votes['annotator_id'].max() + 1

    # 2. Setup Models
    if use_sft:
        policy_model = load_sft_model("gpt2", device)
        ref_model = load_sft_model("gpt2", device)
    else:
        policy_model = DummyCausalLM(vocab_size=50257).to(device)
        ref_model = DummyCausalLM(vocab_size=50257).to(device)
        
    ref_model.eval()
    optimizer = optim.AdamW(policy_model.parameters(), lr=1e-5)

    # 3. Setup EM Integration
    os.makedirs("../../results/joint_tracking", exist_ok=True)
    
    if not use_joint_em:
        # STATIC MODE: Load weights once
        static_weights = pd.read_csv("../../results/04_em_weights.csv")['trust_weight'].values
        trust_weights = torch.tensor(static_weights, dtype=torch.float32, device=device)
    else:
        # JOINT MODE: Initialize EM Engine
        em_engine = DawidSkeneEM(num_prompts, num_annotators)

    metrics_history = []

    # 4. Training Loop
    for epoch in range(epochs):
        if use_joint_em:
            print("--- Joint Update: Running EM with LLM Priors ---")
            llm_priors = get_llm_priors(policy_model, ref_model, train_loader, num_prompts, device, use_sft, beta)
            em_engine.e_step(df_votes, llm_priors=llm_priors)
            em_engine.m_step(df_votes)
            
            # Save epoch-wise EM tracking CSVs
            pd.DataFrame({'annotator_id': range(num_annotators), 'inferred_acc': em_engine.alpha}).to_csv(f"../../results/joint_tracking/em_params_epoch_{epoch+1}.csv", index=False)
            trust_weights = torch.tensor(em_engine.gamma, dtype=torch.float32, device=device)

        policy_model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} DPO Training"):
            ids_A, ids_B = batch['input_ids_A'].to(device), batch['input_ids_B'].to(device)
            p_ids = batch['prompt_id']
            votes = batch['vote'].to(torch.float32).to(device)
            
            # Fetch the trust weight for this specific batch of prompts
            batch_trust = trust_weights[p_ids]
            
            optimizer.zero_grad()
            pi_A = get_batch_logps(get_logits(policy_model, ids_A, use_sft), ids_A)
            pi_B = get_batch_logps(get_logits(policy_model, ids_B, use_sft), ids_B)
            with torch.no_grad():
                ref_A = get_batch_logps(get_logits(ref_model, ids_A, use_sft), ids_A)
                ref_B = get_batch_logps(get_logits(ref_model, ids_B, use_sft), ids_B)
            
            # Projected DPO handles the vote switching naturally using the Trust Weight!
            # If vote is 0 (B), we swap A and B, AND we invert the trust weight (1 - gamma)
            loss = 0
            for i in range(len(votes)):
                if votes[i] == 1:
                    loss += projected_dpo_loss(pi_A[i:i+1], pi_B[i:i+1], ref_A[i:i+1], ref_B[i:i+1], batch_trust[i], beta)
                else:
                    loss += projected_dpo_loss(pi_B[i:i+1], pi_A[i:i+1], ref_B[i:i+1], ref_A[i:i+1], 1.0 - batch_trust[i], beta)
                    
            loss = loss / len(votes)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        test_acc = evaluate_accuracy(policy_model, test_loader, device, use_sft)
        
        print(f"Train Loss: {avg_loss:.4f} | Test Golden Accuracy: {test_acc * 100:.2f}%")
        
        metrics_history.append({"epoch": epoch + 1, "train_loss": avg_loss, "test_accuracy": test_acc})
        pd.DataFrame(metrics_history).to_csv("../../results/projected_metrics.csv", index=False)

    torch.save(policy_model.state_dict(), "../../models/projected_dpo_model.pth")
    print("Projected DPO model saved!")
    # Final Bookkeeping at end of training
    if use_joint_em:
        # Save final inferred params
        df_final_params = pd.DataFrame({'annotator_id': range(num_annotators), 'inferred_acc': em_engine.alpha})
        df_final_params.to_csv("../../results/05_em_inferred_params_joint.csv", index=False)
        
        # Save final trust weights
        df_final_weights = pd.DataFrame({'prompt_id': range(num_prompts), 'trust_weight': em_engine.gamma})
        df_final_weights.to_csv("../../results/04_em_weights_joint.csv", index=False)

if __name__ == "__main__":
    train_projected_dpo(epochs=2, use_sft=False, use_joint_em=True)
    