import random
import numpy as np
import pandas as pd
import os

def simulate_crowd(n_annotators=50, l_workers_per_task=3, 
                   adv_pct=0.1, spam_pct=0.2, exp_pct=0.2,
                   input_file="../../data/processed/01_train_raw.csv", 
                   output_dir="../../data/processed/"):
    
    print("Loading raw training data...")
    df_train = pd.read_csv(input_file)
    target_tasks = len(df_train)
    
    # 1. Math: Force the requested N by calculating R and trimming tasks
    target_edges = target_tasks * l_workers_per_task
    r_tasks_per_worker = target_edges // n_annotators
    
    actual_edges = n_annotators * r_tasks_per_worker
    actual_tasks = actual_edges // l_workers_per_task
    
    # Trim to ensure a perfect regular graph
    df_train = df_train.head(actual_tasks)
    
    print(f"Graph Config: {actual_tasks} Tasks | {n_annotators} Annotators | L={l_workers_per_task} | R={r_tasks_per_worker}")

    # 2. Assign Crowd Demographics based on your percentages
    num_adv = int(n_annotators * adv_pct)
    num_spam = int(n_annotators * spam_pct)
    num_exp = int(n_annotators * exp_pct)
    num_avg = n_annotators - (num_adv + num_spam + num_exp)

    types = (['expert']*num_exp + ['average']*num_avg + 
             ['spammer']*num_spam + ['adversary']*num_adv)
    np.random.shuffle(types)

    true_theta = {}
    for i, a_type in enumerate(types):
        if a_type == 'expert': true_theta[i] = np.random.uniform(0.90, 0.99)
        elif a_type == 'average': true_theta[i] = np.random.uniform(0.65, 0.85)
        elif a_type == 'spammer': true_theta[i] = np.random.uniform(0.45, 0.55)
        elif a_type == 'adversary': true_theta[i] = np.random.uniform(0.05, 0.20)

    # 3. Simulate Voting
    task_stubs = [i for i in df_train['prompt_id'].values for _ in range(l_workers_per_task)]
    worker_stubs = [j for j in range(n_annotators) for _ in range(r_tasks_per_worker)]
    random.shuffle(worker_stubs)
    assignments = list(zip(task_stubs, worker_stubs))

    noisy_data = []
    for prompt_id in df_train['prompt_id'].values:
        truth_is_A = df_train[df_train['prompt_id'] == prompt_id]['truth_is_A'].values[0]
        assigned_workers = [w for t, w in assignments if t == prompt_id]

        for worker_id in assigned_workers:
            picked_true = random.random() < true_theta[worker_id]
            vote = 1 if (picked_true == truth_is_A) else 0
            noisy_data.append({"prompt_id": prompt_id, "annotator_id": worker_id, "vote": vote})

    pd.DataFrame(noisy_data).to_csv(f"{output_dir}02_train_noisy_votes.csv", index=False)
    true_params_path = "../../data/true_params/03_true_params.csv"
    os.makedirs(os.path.dirname(true_params_path), exist_ok=True)
    pd.DataFrame([{"annotator_id": k, "true_acc": v} for k, v in true_theta.items()]).to_csv(true_params_path, index=False)
    print(f"Saved 02_train_noisy_votes.csv and 03_true_params.csv to {output_dir}")

if __name__ == "__main__":
    simulate_crowd()