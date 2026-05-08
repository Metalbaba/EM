import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class DawidSkeneEM:
    def __init__(self, num_prompts, num_annotators):
        self.num_prompts = num_prompts
        self.num_annotators = num_annotators
        
        # Initialize priors
        # gamma: Probability that Response A is the true winner for prompt i
        self.gamma = np.full(num_prompts, 0.5)
        
        # alpha: Probability that annotator j answers correctly
        # Initialize slightly above 0.5 to break symmetry
        self.alpha = np.full(num_annotators, 0.6) 

    def m_step(self, df_votes):
        """
        Maximization Step: Update annotator accuracy (alpha) based on current belief of truth (gamma).
        """
        correct_votes = np.zeros(self.num_annotators)
        total_votes = np.zeros(self.num_annotators)
        
        for _, row in df_votes.iterrows():
            p_id = int(row['prompt_id'])
            a_id = int(row['annotator_id'])
            vote = int(row['vote']) # 1 for A, 0 for B
            
            # If vote is A, they are correct if truth is A (gamma).
            # If vote is B, they are correct if truth is B (1 - gamma).
            prob_correct = self.gamma[p_id] if vote == 1 else (1.0 - self.gamma[p_id])
            
            correct_votes[a_id] += prob_correct
            total_votes[a_id] += 1.0
            
        # Update alpha, avoiding division by zero
        # Add a tiny epsilon to prevent alpha from hitting exactly 0 or 1
        epsilon = 1e-4
        self.alpha = np.clip(correct_votes / np.maximum(total_votes, 1), epsilon, 1.0 - epsilon)

    def e_step(self, df_votes, llm_priors=None):
        """
        Expectation Step: Update belief of truth (gamma) based on annotator accuracy (alpha).
        """
        # Start with a flat 50/50 prior, or the LLM's prior if provided
        log_prob_A = np.log(llm_priors) if llm_priors is not None else np.full(self.num_prompts, np.log(0.5))
        log_prob_B = np.log(1.0 - llm_priors) if llm_priors is not None else np.full(self.num_prompts, np.log(0.5))
        
        for _, row in df_votes.iterrows():
            p_id = int(row['prompt_id'])
            a_id = int(row['annotator_id'])
            vote = int(row['vote'])
            
            # If true answer is A:
            prob_vote_given_A = self.alpha[a_id] if vote == 1 else (1.0 - self.alpha[a_id])
            log_prob_A[p_id] += np.log(prob_vote_given_A)
            
            # If true answer is B:
            prob_vote_given_B = (1.0 - self.alpha[a_id]) if vote == 1 else self.alpha[a_id]
            log_prob_B[p_id] += np.log(prob_vote_given_B)
            
        # Normalize back to probabilities using log-sum-exp trick for numerical stability
        max_log = np.maximum(log_prob_A, log_prob_B)
        prob_A = np.exp(log_prob_A - max_log)
        prob_B = np.exp(log_prob_B - max_log)
        
        self.gamma = prob_A / (prob_A + prob_B)

def run_standalone_em(max_iterations=20, tolerance=1e-4):
    print("Loading data for EM Optimization...")
    df_votes = pd.read_csv("../../data/processed/02_train_noisy_votes.csv")
    df_true_params = pd.read_csv("../../data/true_params/03_true_params.csv")
    # df_true_params = pd.read_csv("../../data/processed/03_true_params.csv")
    
    num_prompts = df_votes['prompt_id'].max() + 1
    num_annotators = df_votes['annotator_id'].max() + 1
    
    em = DawidSkeneEM(num_prompts, num_annotators)
    
    loss_history = []
    
    print(f"Running Dawid-Skene EM for {max_iterations} iterations...")
    for iteration in range(max_iterations):
        old_gamma = em.gamma.copy()
        
        # The EM Loop
        em.e_step(df_votes)
        em.m_step(df_votes)
        # Ensure directory exists and save frame-by-frame tracking for animation
        os.makedirs("../../results/standalone_tracking", exist_ok=True)
        pd.DataFrame({'annotator_id': range(num_annotators), 'inferred_acc': em.alpha}).to_csv(f"../../results/standalone_tracking/em_params_iter_{iteration+1}.csv", index=False)
        # Calculate Convergence (How much did gamma change?)
        gamma_shift = np.max(np.abs(em.gamma - old_gamma))
        
        # Calculate Parameter L1 Loss (How close are we to the hidden truth?)
        # We merge our current alpha estimates with the true parameters
        df_current_alpha = pd.DataFrame({'annotator_id': range(num_annotators), 'est_acc': em.alpha})
        df_merged = pd.merge(df_true_params, df_current_alpha, on='annotator_id')
        l1_loss = np.mean(np.abs(df_merged['true_acc'] - df_merged['est_acc']))
        
        loss_history.append({"iteration": iteration + 1, "l1_parameter_loss": l1_loss, "max_gamma_shift": gamma_shift})
        
        print(f"Iter {iteration+1:02d} | L1 Param Loss: {l1_loss:.4f} | Max Shift: {gamma_shift:.6f}")
        
        if gamma_shift < tolerance:
            print(f"Converged early at iteration {iteration+1}!")
            break

    # Save Output Checkpoints
    os.makedirs("../../results", exist_ok=True)
    
    # 1. Save Loss History
    pd.DataFrame(loss_history).to_csv("../../results/em_loss_history.csv", index=False)
    
    # 2. Save Inferred Annotator Parameters
    df_inferred = pd.DataFrame({'annotator_id': range(num_annotators), 'inferred_acc': em.alpha})
    df_inferred.to_csv("../../results/05_em_inferred_params.csv", index=False)
    
    # 3. Save the Trust Weights (gamma) for the LLM
    df_weights = pd.DataFrame({'prompt_id': range(num_prompts), 'trust_weight': em.gamma})
    df_weights.to_csv("../../results/04_em_weights.csv", index=False)
    
    print("EM Optimization Complete! Checkpoints saved to results/")

if __name__ == "__main__":
    run_standalone_em()