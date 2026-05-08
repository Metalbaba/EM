import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_stress_results():
    data_path = "../../results/stress_tests/phase3_sweep_matrix.csv"
    output_dir = "../../results/stress_tests/"
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Run the sweep script first!")
        return

    # Load the sweep data
    df = pd.read_csv(data_path)
    
    # ==========================================
    # 1. Plot the Sparsity vs. Adversary Heatmap
    # ==========================================
    print("Generating Sparsity vs. Adversary Heatmap...")
    df_heatmap = df[df['variable_tested'] == 'Heatmap'].copy()
    
    # Pivot the dataframe into a 2D grid
    # index (Y-axis): L (Sparsity)
    # columns (X-axis): Adv (Adversarial Percentage)
    # values: Error (L1 Parameter Error)
    pivot_table = df_heatmap.pivot(index='L', columns='Adv', values='Error')
    
    # Sort L descending so the "safest" dense graphs are at the top
    pivot_table = pivot_table.sort_index(ascending=False)
    
    # Convert adversarial decimals to percentages for cleaner labels
    pivot_table.columns = [f"{int(col*100)}%" for col in pivot_table.columns]
    
    plt.figure(figsize=(10, 8))
    # cmap='YlOrRd' makes low error Yellow (Safe) and high error Red (Danger)
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlOrRd", 
                cbar_kws={'label': 'L1 Parameter Error (Lower is Better)'},
                linewidths=.5)
    
    plt.title("EM Algorithm 'Safe Zone' Heatmap\n(Sparsity vs. Malicious Intent)", fontsize=16, pad=15)
    plt.ylabel("Workers per Task ($L$)", fontsize=12)
    plt.xlabel("Adversary Percentage", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_L_vs_Adv.png"), dpi=300)
    plt.show()

    # ==========================================
    # 2. Plot the Scale Robustness Line Graph
    # ==========================================
    print("Generating Scale Robustness Line Graph...")
    df_scale = df[df['variable_tested'] == 'Scale'].copy()
    
    # If the scale test didn't run or is empty, skip this part
    if not df_scale.empty:
        plt.figure(figsize=(8, 6))
        
        plt.plot(df_scale['N'], df_scale['Error'], marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
        
        # Add a baseline threshold line where EM is considered "broken"
        plt.axhline(y=0.15, color='r', linestyle='--', alpha=0.7, label='Failure Threshold')
        
        plt.title("EM Robustness vs. Crowd Size ($N$)\n(Fixed $L=3$, Adv=10%)", fontsize=16, pad=15)
        plt.xlabel("Number of Annotators ($N$)", fontsize=12)
        plt.ylabel("L1 Parameter Error", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "linegraph_scale_N.png"), dpi=300)
        plt.show()

if __name__ == "__main__":
    plot_stress_results()