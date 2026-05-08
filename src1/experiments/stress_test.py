import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# Ensure Python can find your existing scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.simulate_crowd import simulate_crowd
from models.em_standalone import DawidSkeneEM

def calculate_l1_loss(em_engine, true_params_path):
    df_true = pd.read_csv(true_params_path)
    df_est = pd.DataFrame({'annotator_id': range(len(em_engine.alpha)), 'est_acc': em_engine.alpha})
    df_merged = pd.merge(df_true, df_est, on='annotator_id')
    return np.mean(np.abs(df_merged['true_acc'] - df_merged['est_acc']))

def run_single_config(L, adv_pct, N, track_frames=False, case_name=""):
    """Runs a single simulation and EM optimization. Returns the L1 Error."""
    # Ensure spammers adjust dynamically so experts remain at ~20%
    spam_pct = max(0.0, 0.40 - adv_pct) 
    
    if track_frames:
        print(f"\n[Generating Animation Frames for: {case_name}]")
        data_dir = f"../../data/processed/edge_cases/{case_name}/"
        tracking_dir = f"../../results/edge_cases/{case_name}_tracking/"
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(tracking_dir, exist_ok=True)
    else:
        data_dir = "../../data/processed/"
        tracking_dir = None

    # 1. Simulate the Crowd (Modular: automatically scales to whatever size 01_train_raw.csv is)
    simulate_crowd(
        n_annotators=N, 
        l_workers_per_task=L, 
        adv_pct=adv_pct, 
        spam_pct=spam_pct, 
        exp_pct=0.2,
        input_file="../../data/processed/01_train_raw.csv",
        output_dir=data_dir
    )
    
    # 2. Run EM
    df_votes = pd.read_csv(f"{data_dir}02_train_noisy_votes.csv")
    num_prompts = df_votes['prompt_id'].max() + 1
    
    em = DawidSkeneEM(num_prompts, N)
    
    for iteration in range(20):
        old_gamma = em.gamma.copy()
        em.e_step(df_votes)
        em.m_step(df_votes)
        
        # Save tracking frame if requested
        if track_frames:
            df_frame = pd.DataFrame({'annotator_id': range(N), 'inferred_acc': em.alpha})
            df_frame.to_csv(f"{tracking_dir}em_params_iter_{iteration+1}.csv", index=False)
            
        if np.max(np.abs(em.gamma - old_gamma)) < 1e-4:
            break
            
    # 3. Calculate Error
    true_params_path = "../../data/true_params/03_true_params.csv"
    l1_loss = calculate_l1_loss(em, true_params_path)
    return l1_loss

def run_stress_pipeline():
    os.makedirs("../../results/stress_tests", exist_ok=True)
    
    # --- PHASE 1: THE SWEEP ---
    print("=== PHASE 1: Sweeping Parameter Space ===")
    results = []
    
    # Standard baseline we test against
    BASE_N = 100
    BASE_L = 3
    BASE_ADV = 0.10
    ERROR_THRESHOLD = 0.15 # The point where EM is considered "broken"
    
    # Define Sweep Grids
    L_grid = [1, 2, 3, 5]
    Adv_grid = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
    N_grid = [20, 50, 100, 200, 500]
    
    total_runs = (len(L_grid) * len(Adv_grid)) + len(N_grid)
    
    with tqdm(total=total_runs, desc="Simulating") as pbar:
        # Sweep A: Heatmap Matrix (Sparsity vs Adversaries)
        for L in L_grid:
            for adv in Adv_grid:
                loss = run_single_config(L, adv, BASE_N)
                results.append({"variable_tested": "Heatmap", "L": L, "Adv": adv, "N": BASE_N, "Error": loss})
                pbar.update(1)
                
        # Sweep B: Scale/Density (Annotators)
        for N in N_grid:
            loss = run_single_config(BASE_L, BASE_ADV, N)
            results.append({"variable_tested": "Scale", "L": BASE_L, "Adv": BASE_ADV, "N": N, "Error": loss})
            pbar.update(1)
            
    df_results = pd.DataFrame(results)
    df_results.to_csv("../../results/stress_tests/phase3_sweep_matrix.csv", index=False)
    print("Sweep Matrix saved to results/stress_tests/phase3_sweep_matrix.csv")
    
    # --- PHASE 2: AUTOMATIC EDGE CASE DETECTION ---
    print("\n=== PHASE 2: Detecting Independent Failure Modes ===")
    
    bad_L = df_results[(df_results['Adv'] == BASE_ADV) & (df_results['N'] == BASE_N) & (df_results['Error'] > ERROR_THRESHOLD)]
    bad_Adv = df_results[(df_results['L'] == BASE_L) & (df_results['N'] == BASE_N) & (df_results['Error'] > ERROR_THRESHOLD)]
    bad_N = df_results[(df_results['variable_tested'] == "Scale") & (df_results['Error'] > ERROR_THRESHOLD)]
    
    # Select the breaking points (or default to extremes if it never technically broke)
    edge_L = bad_L['L'].max() if not bad_L.empty else L_grid[0] 
    edge_Adv = bad_Adv['Adv'].min() if not bad_Adv.empty else Adv_grid[-1]
    edge_N = bad_N['N'].min() if not bad_N.empty else N_grid[0]
    
    print(f"Isolated Failure due to Sparsity (L): L={edge_L}")
    print(f"Isolated Failure due to Adversaries: Adv={edge_Adv*100}%")
    print(f"Isolated Failure due to Scale (N): N={edge_N}")
    
    # --- PHASE 3: GENERATE ANIMATION FRAMES ---
    print("\n=== PHASE 3: Generating Animation Frames for Edge Cases ===")
    
    run_single_config(L=edge_L, adv_pct=BASE_ADV, N=BASE_N, track_frames=True, case_name="isolated_sparsity_failure")
    run_single_config(L=BASE_L, adv_pct=edge_Adv, N=BASE_N, track_frames=True, case_name="isolated_adversarial_failure")
    run_single_config(L=BASE_L, adv_pct=BASE_ADV, N=edge_N, track_frames=True, case_name="isolated_scale_failure")
    
    # And one Golden Standard for baseline comparison
    run_single_config(L=5, adv_pct=0.05, N=200, track_frames=True, case_name="golden_standard")
    
    print("\nPhase 3 Complete! You can now animate these using animate_clusters.py")

if __name__ == "__main__":
    run_stress_pipeline()