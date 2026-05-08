import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.simulate_crowd import simulate_crowd
from models.em_standalone import DawidSkeneEM

def calculate_l1_loss(em_engine, true_params_path):
    df_true = pd.read_csv(true_params_path)
    df_est = pd.DataFrame({'annotator_id': range(len(em_engine.alpha)), 'est_acc': em_engine.alpha})
    df_merged = pd.merge(df_true, df_est, on='annotator_id')
    return np.mean(np.abs(df_merged['true_acc'] - df_merged['est_acc']))

def run_single_config(L, adv_pct, N, track_frames=False, case_name=""):
    spam_pct = max(0.0, 0.40 - adv_pct) 
    
    # FIX: Dynamically assign paths so we NEVER overwrite the main Phase 2 data
    if track_frames:
        data_dir = f"../../data/processed/edge_cases/{case_name}/"
        tracking_dir = f"../../results/edge_cases/{case_name}_tracking/"
    else:
        # Isolated Temp directory for rapid heatmap sweeps
        data_dir = "../../data/processed/sweep_temp/"
        tracking_dir = None
        
    os.makedirs(data_dir, exist_ok=True)
    if tracking_dir: os.makedirs(tracking_dir, exist_ok=True)

    simulate_crowd(
        n_annotators=N, 
        l_workers_per_task=L, 
        adv_pct=adv_pct, 
        spam_pct=spam_pct, 
        exp_pct=0.2,
        input_file="../../data/processed/01_train_raw.csv",
        output_dir=data_dir
    )
    
    df_votes = pd.read_csv(f"{data_dir}02_train_noisy_votes.csv")
    num_prompts = df_votes['prompt_id'].max() + 1
    
    em = DawidSkeneEM(num_prompts, N)
    
    for iteration in range(20):
        old_gamma = em.gamma.copy()
        em.e_step(df_votes)
        em.m_step(df_votes)
        
        if track_frames:
            df_frame = pd.DataFrame({'annotator_id': range(N), 'inferred_acc': em.alpha})
            df_frame.to_csv(f"{tracking_dir}em_params_iter_{iteration+1}.csv", index=False)
            
        if np.max(np.abs(em.gamma - old_gamma)) < 1e-4:
            break
            
    # FIX: Must read the true parameters from the exact same isolated data directory
    true_params_path = f"{data_dir}03_true_params.csv"
    l1_loss = calculate_l1_loss(em, true_params_path)
    return l1_loss

def run_stress_pipeline():
    os.makedirs("../../results/stress_tests", exist_ok=True)
    
    print("=== PHASE 1: Sweeping Parameter Space ===")
    results = []
    BASE_N, BASE_L, BASE_ADV = 400, 3, 0.10
    ERROR_THRESHOLD = 0.15 
    
    L_grid = [1, 2, 3, 5]
    Adv_grid = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
    N_grid = [50, 100, 200, 400, 800]
    
    total_runs = (len(L_grid) * len(Adv_grid)) + len(N_grid)
    
    with tqdm(total=total_runs, desc="Simulating") as pbar:
        for L in L_grid:
            for adv in Adv_grid:
                loss = run_single_config(L, adv, BASE_N)
                results.append({"variable_tested": "Heatmap", "L": L, "Adv": adv, "N": BASE_N, "Error": loss})
                pbar.update(1)
                
        for N in N_grid:
            loss = run_single_config(BASE_L, BASE_ADV, N)
            results.append({"variable_tested": "Scale", "L": BASE_L, "Adv": BASE_ADV, "N": N, "Error": loss})
            pbar.update(1)
            
    df_results = pd.DataFrame(results)
    df_results.to_csv("../../results/stress_tests/phase3_sweep_matrix.csv", index=False)
    
    print("\n=== PHASE 2: Detecting Failure Modes ===")
    bad_L = df_results[(df_results['Adv'] == BASE_ADV) & (df_results['N'] == BASE_N) & (df_results['Error'] > ERROR_THRESHOLD)]
    bad_Adv = df_results[(df_results['L'] == BASE_L) & (df_results['N'] == BASE_N) & (df_results['Error'] > ERROR_THRESHOLD)]
    bad_N = df_results[(df_results['variable_tested'] == "Scale") & (df_results['Error'] > ERROR_THRESHOLD)]
    
    edge_L = bad_L['L'].max() if not bad_L.empty else L_grid[0] 
    edge_Adv = bad_Adv['Adv'].min() if not bad_Adv.empty else Adv_grid[-1]
    edge_N = bad_N['N'].min() if not bad_N.empty else N_grid[0]
    
    print("\n=== PHASE 3: Generating Animation Frames ===")
    run_single_config(L=edge_L, adv_pct=BASE_ADV, N=BASE_N, track_frames=True, case_name="isolated_sparsity_failure")
    run_single_config(L=BASE_L, adv_pct=edge_Adv, N=BASE_N, track_frames=True, case_name="isolated_adversarial_failure")
    run_single_config(L=BASE_L, adv_pct=BASE_ADV, N=edge_N, track_frames=True, case_name="isolated_scale_failure")
    run_single_config(L=5, adv_pct=0.05, N=200, track_frames=True, case_name="golden_standard")
    
    print("\nPhase 3 Complete!")

if __name__ == "__main__":
    run_stress_pipeline()