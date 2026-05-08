import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_training_metrics(base_model="gpt2"):
    """
    Finds all metric CSVs in the results folder and plots them on a single comparative graph.
    base_model: "gpt2" or "dummy" to filter which models to plot.
    """
    results_dir = "../../results/"
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} not found.")
        return
        
    # Find all metric files for the specified base model
    pattern = os.path.join(results_dir, f"*_{base_model}_metrics.csv")
    metric_files = glob.glob(pattern)
    
    if not metric_files:
        print(f"No metric CSVs found for model type: {base_model.upper()}")
        print("Run your training scripts first!")
        return
        
    print(f"Found {len(metric_files)} metric files. Generating Climax Graphs...")
    
    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Color palette to make the thesis graph highly readable
    color_map = {
        "oracle": ("black", "--", "Oracle (Perfect Ground Truth)"),
        "baseline": ("red", "-", "Baseline (Noisy Spammers)"),
        "projected_gpt2_static": ("blue", "-", "Static EM-DPO"),
        "projected_gpt2_joint": ("green", "-", "Joint EM-DPO"),
        "projected_dummy_static": ("blue", "-", "Static EM-DPO"),
        "projected_dummy_joint": ("green", "-", "Joint EM-DPO")
    }
    
    for file in metric_files:
        filename = os.path.basename(file)
        name_key = filename.replace("_metrics.csv", "")
        
        # Try to match our predefined styling, otherwise use grey
        color, linestyle, label = "gray", "-", name_key
        for key, styling in color_map.items():
            if key in name_key:
                color, linestyle, label = styling
                break
        
        try:
            df = pd.read_csv(file)
            if 'epoch' not in df.columns:
                continue
                
            # Plot Training Loss
            ax1.plot(df['epoch'], df['train_loss'], label=label, color=color, linestyle=linestyle, marker='o', linewidth=2)
            
            # Plot Test Accuracy (Convert to percentage)
            ax2.plot(df['epoch'], df['test_accuracy'] * 100, label=label, color=color, linestyle=linestyle, marker='s', linewidth=2)
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            
    # Customize Loss Plot
    ax1.set_title(f"Training Loss over Epochs ({base_model.upper()})", fontsize=15, pad=10)
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("DPO Loss", fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11)
    
    # Customize Accuracy Plot
    ax2.set_title(f"Test Accuracy over Epochs ({base_model.upper()})", fontsize=15, pad=10)
    ax2.set_xlabel("Epoch", fontsize=13)
    ax2.set_ylabel("Golden Accuracy (%)", fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, f"climax_training_graphs_{base_model}.png")
    plt.savefig(output_path, dpi=300)
    plt.show()
    
    print(f"Climax graph saved successfully to: {output_path}")

if __name__ == "__main__":
    # Change to "dummy" if you are plotting your local Mac CPU runs
    plot_training_metrics("gpt2")