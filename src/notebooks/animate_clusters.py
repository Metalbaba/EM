import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import re
import argparse

def animate_em_clusters(tracking_dir, title_suffix="", true_params_path=""):
    """Animates the EM parameter convergence from a specific directory."""
    
    if not os.path.exists(true_params_path):
        print(f"Error: Ground truth file not found at {true_params_path}")
        return
    
    # 1. Check Paths
    # df_true_path = "../../data/true_params/03_true_params.csv"
    df_true_path = true_params_path
    if not os.path.exists(df_true_path):
        print(f"Error: Ground truth file not found at {df_true_path}")
        return

    if not os.path.exists(tracking_dir):
        print(f"Error: Tracking directory '{tracking_dir}' not found.")
        print("Did you run the training or edge case script yet?")
        return

    # 2. Get and Sort Files Naturally (e.g., iter_2 before iter_10)
    epoch_files = [f for f in os.listdir(tracking_dir) if f.endswith('.csv')]
    if not epoch_files:
        print(f"Error: No CSV tracking frames found in '{tracking_dir}'.")
        return
        
    epoch_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    num_frames = len(epoch_files)

    # 3. Load True Parameters
    df_true = pd.read_csv(df_true_path)
    true_accs = df_true['true_acc'].values
    num_annotators = len(true_accs)
    
    # 4. Setup Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Estimation (y=x)')
    ax.set_xlabel('True Annotator Accuracy ($p$)')
    ax.set_ylabel('EM Inferred Accuracy ($\\alpha$)')
    ax.set_title(f'EM Parameter Evolution {title_suffix}')
    
    # Initialize scatter with the standard 0.6 prior
    scatter = ax.scatter(true_accs, [0.6]*num_annotators, c=true_accs, cmap='coolwarm_r', s=100, edgecolors='k')
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # 5. Animation Function
    def update(frame):
        df_frame = pd.read_csv(os.path.join(tracking_dir, epoch_files[frame]))
        inferred_accs = df_frame['inferred_acc'].values
        
        scatter.set_offsets(list(zip(true_accs, inferred_accs)))
        time_text.set_text(f'Frame: {frame + 1} / {num_frames}')
        return scatter, time_text

    print(f"Playing animation from {num_frames} frames...")
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=400, blit=True, repeat_delay=2000)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate EM Clustering over time.")
    
    # Allow the user to specify standard modes OR a custom path
    parser.add_argument('--mode', type=str, choices=['standalone', 'joint'], 
                        help="Animate the standard Phase 2 pipelines.")
    parser.add_argument('--path', type=str, 
                        help="Direct path to a tracking folder (e.g., for Phase 3 Edge Cases).")
    
    args = parser.parse_args()

    if args.path:
        folder_name = os.path.basename(os.path.normpath(args.path))
        norm = args.path.replace("\\", "/")
        if "edge_cases" in norm:
            base_name = folder_name.replace("_tracking", "")
            true_params = f"../../data/processed/edge_cases/{base_name}/03_true_params.csv"
        else:
            # Main pipeline / projected DPO: worker truth lives next to default votes
            true_params = "../../data/processed/03_true_params.csv"
        animate_em_clusters(args.path, title_suffix=f"[{folder_name}]", true_params_path=true_params)

    elif args.mode:
        target_dir = f"../../results/{args.mode}_tracking/"
        true_params = "../../data/processed/03_true_params.csv"
        animate_em_clusters(target_dir, title_suffix=f"[{args.mode.capitalize()} Mode]", true_params_path=true_params)
        
    else:
        print("Please provide a target to animate.")
        print("Example 1: python animate_clusters.py --mode standalone")
        print("Example 2: python animate_clusters.py --mode joint")
        print("Example 3: python animate_clusters.py --path ../../results/edge_cases/02_sparsity_crisis_tracking/")