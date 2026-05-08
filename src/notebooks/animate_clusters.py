import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import re

def animate_em_clusters(mode="standalone"):
    """
    mode: "standalone" (looks in results/standalone_tracking/)
          "joint"      (looks in results/joint_tracking/)
    """
    # 1. Check Paths
    df_true_path = "../../data/true_params/03_true_params.csv"
    if not os.path.exists(df_true_path):
        print(f"Error: Ground truth file not found at {df_true_path}")
        return

    tracking_dir = f"../../results/{mode}_tracking/"
    if not os.path.exists(tracking_dir):
        print(f"Error: Directory '{tracking_dir}' not found. Did you run the {mode} script?")
        return

    # 2. Get and Sort Files Naturally (1, 2, ..., 10, 11)
    epoch_files = [f for f in os.listdir(tracking_dir) if f.endswith('.csv')]
    if not epoch_files:
        print(f"Error: No CSVs found in '{tracking_dir}'.")
        return
        
    epoch_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    num_epochs = len(epoch_files)

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
    ax.set_title(f'EM Parameter Evolution ({mode.capitalize()} Mode)')
    
    # Initialize scatter with 0.6 prior
    scatter = ax.scatter(true_accs, [0.6]*num_annotators, c=true_accs, cmap='coolwarm_r', s=100, edgecolors='k')
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=14, fontweight='bold')

    # 5. Animation Function
    def update(frame):
        df_frame = pd.read_csv(os.path.join(tracking_dir, epoch_files[frame]))
        inferred_accs = df_frame['inferred_acc'].values
        
        scatter.set_offsets(list(zip(true_accs, inferred_accs)))
        label_type = "Epoch" if mode == "joint" else "Iteration"
        time_text.set_text(f'{label_type}: {frame + 1}')
        return scatter, time_text

    ani = animation.FuncAnimation(fig, update, frames=num_epochs, interval=500, blit=True, repeat_delay=2000)
    plt.show()

if __name__ == "__main__":
    # Change to "joint" when testing the Joint DPO loop
    animate_em_clusters(mode="standalone")