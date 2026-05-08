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
from models.em_standalone import DawidSkeneEM

def get_device():
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    return "cpu"

def get_logits(model, input_ids, use_sft):
    return model(input_ids).logits if use_sft else model(input_ids)

def get_llm_priors(policy_model, ref_model, dataloader, num_prompts, device, use_sft, beta=0.1):
    policy_model.eval()
    priors = np.full(num_prompts, 0.5)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Priors", leave=False):
            ids_A, ids_B = batch['input_ids_A'].to(device), batch['input_ids_B'].to(device)
            p_ids = batch['prompt_id'].numpy()
            pi_A = get_batch_logps(get_logits(policy_model, ids_A, use_sft), ids_A)
            pi_B = get_batch_logps(get_logits(policy_model, ids_B, use_sft), ids_B)
            ref_A = get_batch_logps(get_logits(ref_model, ids_A, use_sft), ids_A)
            ref_B = get_batch_logps(get_logits(ref_model, ids_B, use_sft), ids_B)
            
            logits = (pi_A - pi_B) - (ref_A - ref_B)
            prob_A = torch.sigmoid(beta * logits).cpu().numpy()
            for i, p_id in enumerate(p_ids): priors[p_id] = prob_A[i]
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
    base_name = "gpt2" if use_sft else "dummy"
    em_mode = "joint" if use_joint_em else "static"
    print(f"Projected DPO | Mode: {em_mode.upper()} | Model: {base_name.upper()} | Device: {device}")

    # 1. Load Noisy Data
    train_dataset = torch.load("../../data/tokenized/noisy_train_tokens.pt")
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

    # 3. Setup Dynamic File Paths
    csv_filename = f"../../results/projected_{base_name}_{em_mode}_metrics.csv"
    model_filename = f"../../models/projected_{base_name}_{em_mode}.pth"
    tracking_dir = f"../../results/{em_mode}_tracking_{base_name}/"
    os.makedirs(tracking_dir, exist_ok=True)
    
    if not use_joint_em:
        static_weights = pd.read_csv("../../results/04_em_weights.csv")['trust_weight'].values
        trust_weights = torch.tensor(static_weights, dtype=torch.float32, device=device)
    else:
        em_engine = DawidSkeneEM(num_prompts, num_annotators)

    metrics_history = []

    # 4. Training Loop
    for epoch in range(epochs):
        if use_joint_em:
            llm_priors = get_llm_priors(policy_model, ref_model, train_loader, num_prompts, device, use_sft, beta)
            em_engine.e_step(df_votes, llm_priors=llm_priors)
            em_engine.m_step(df_votes)
            pd.DataFrame({'annotator_id': range(num_annotators), 'inferred_acc': em_engine.alpha}).to_csv(f"{tracking_dir}em_params_epoch_{epoch+1}.csv", index=False)
            trust_weights = torch.tensor(em_engine.gamma, dtype=torch.float32, device=device)

        policy_model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            ids_A, ids_B = batch['input_ids_A'].to(device), batch['input_ids_B'].to(device)
            p_ids = batch['prompt_id']
            votes = batch['vote'].to(torch.float32).to(device)
            batch_trust = trust_weights[p_ids]
            
            optimizer.zero_grad()
            pi_A = get_batch_logps(get_logits(policy_model, ids_A, use_sft), ids_A)
            pi_B = get_batch_logps(get_logits(policy_model, ids_B, use_sft), ids_B)
            with torch.no_grad():
                ref_A = get_batch_logps(get_logits(ref_model, ids_A, use_sft), ids_A)
                ref_B = get_batch_logps(get_logits(ref_model, ids_B, use_sft), ids_B)
            
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
        print(f"Train Loss: {avg_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")
        
        metrics_history.append({"epoch": epoch + 1, "train_loss": avg_loss, "test_accuracy": test_acc})
        pd.DataFrame(metrics_history).to_csv(csv_filename, index=False)

    torch.save(policy_model.state_dict(), model_filename)
    
    if use_joint_em:
        pd.DataFrame({'annotator_id': range(num_annotators), 'inferred_acc': em_engine.alpha}).to_csv(f"../../results/05_em_inferred_params_joint_{base_name}.csv", index=False)
        pd.DataFrame({'prompt_id': range(num_prompts), 'trust_weight': em_engine.gamma}).to_csv(f"../../results/04_em_weights_joint_{base_name}.csv", index=False)
    
    print(f"Saved: {model_filename} and {csv_filename}")

if __name__ == "__main__":
    train_projected_dpo(epochs=1, use_sft=False, use_joint_em=False) # Static Dummy