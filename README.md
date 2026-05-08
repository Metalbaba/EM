# Running instructions

Scripts use paths such as `../../data/` relative to the **current working directory**, not the script file. Run each command **from the directory shown** so outputs land under `data/`, `results/`, and `models/` at the repo root.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

For `ground_truth.py` you need network access to download **Anthropic/hh-rlhf** via Hugging Face (`datasets`). For plotting and animations, install **matplotlib** and **seaborn** if they are not already present:

```bash
pip install matplotlib seaborn
```

---

## Phase 1 — Data, tokenization, baseline DPO

**1. Ground truth (HH-RLHF subsample + random A/B)**  
Load Anthropic/hh-rlhf, shuffle each official split with fixed seeds, then keep up to **10,000** train rows from `split="train"` and **2,000** test rows from `split="test"` (fewer if the split is shorter). Each row gets a reproducible random `truth_is_A` draw.

```bash
cd src/data
python ground_truth.py
```

Outputs: `data/processed/01_train_raw.csv`, `data/processed/01_test_raw.csv`

**2. Noisy crowd votes**  
Simulate annotators on that ground truth (graph size, sparsity `L`, adversary + spammer mix, etc.). Produces votes for EM or DPO and saves each worker’s simulated accuracy for evaluation.

```bash
cd src/data
python simulate_crowd.py
```

Outputs: `data/processed/02_train_noisy_votes.csv`, `data/true_params/03_true_params.csv`

**3a. Tokenize noisy train + test (for crowd / EM training)**  
Expands each vote row into token tensors for DPO training on noisy labels.

```bash
cd src/data
python tokenize_data.py
```

Outputs: `data/tokenized/noisy_train_tokens.pt`, `data/tokenized/test_tokens.pt`

**3b. Tokenize oracle train (ceiling / upper bound)**  
Builds one training example per prompt using **`truth_is_A` as the vote** (always agrees with the simulator’s hidden label). Use this with **baseline DPO** and `is_oracle=True` to estimate how high test accuracy can get under the same model and objective, without crowd noise.

```bash
cd src/data
python tokenize_oracle.py
```

Outputs: `data/tokenized/oracle_train_tokens.pt`  
Requires: step **1** only (no noisy votes file needed).

**4. Baseline DPO**  
`train_baseline_dpo.py` loads either **`noisy_train_tokens.pt`** (real crowd) or **`oracle_train_tokens.pt`** (ground-truth preferences). Training uses **one DPO term per row**; if `vote == 0`, A and B are swapped for that row.

Edit `if __name__ == "__main__"` to choose:

- **`use_sft`**: `True` = GPT-2, `False` = small dummy LM  
- **`is_oracle`**: `False` = noisy crowd (`noisy_train_tokens.pt`), `True` = oracle ceiling (`oracle_train_tokens.pt`)  
- **`epochs`**: as needed  

```bash
cd src/training
python train_baseline_dpo.py
```

Outputs (pattern):

| Setting | Metrics CSV | Weights |
|--------|-------------|---------|
| Noisy + dummy | `results/baseline_dummy_metrics.csv` | `models/baseline_dummy.pth` |
| Noisy + GPT-2 | `results/baseline_gpt2_metrics.csv` | `models/baseline_gpt2.pth` |
| Oracle + dummy | `results/oracle_dummy_metrics.csv` | `models/oracle_dummy.pth` |
| Oracle + GPT-2 | `results/oracle_gpt2_metrics.csv` | `models/oracle_gpt2.pth` |

Compare **oracle** vs **baseline** test accuracy to see how much performance is left on the table after accounting for noise and limited training.

**5. Plots**  
Use CSV columns `epoch`, `train_loss`, `test_accuracy` to compare runs.

---

## Phase 2 — EM and projected DPO

Steps **1**, **2**, and **3a** (noisy tokenization) are required. Oracle tokenization is optional here.

**4a. Standalone EM (offline weights)**  
Runs a **binary crowd EM** (per-worker accuracy `alpha`, `em_standalone.py`) on `02_train_noisy_votes.csv` and writes trust weights for **static** projected DPO.

```bash
cd src/models
python em_standalone.py
```

Outputs: `results/standalone_tracking/em_params_iter_*.csv`, `results/04_em_weights.csv`, `results/05_em_inferred_params.csv`, `results/em_loss_history.csv`

**4b. Projected DPO**  
Loads **`noisy_train_tokens.pt`** and `02_train_noisy_votes.csv`. Soft-label DPO with EM’s \(\gamma\) (trust weights). Edit `if __name__ == "__main__"` in `train_projected_dpo.py`:

- **`use_joint_em=True`**: each epoch recomputes LLM priors, runs EM, then trains (no prior `04_em_weights.csv` needed).  
- **`use_joint_em=False`**: static weights from **`results/04_em_weights.csv`** — run **4a** first.  
- **`use_sft`**: GPT-2 vs dummy.

```bash
cd src/training
python train_projected_dpo.py
```

Outputs (pattern):

- Metrics: `results/projected_{gpt2|dummy}_{joint|static}_metrics.csv`  
- Weights: `models/projected_{gpt2|dummy}_{joint|static}.pth`  
- Joint EM only: `results/05_em_inferred_params_joint_{gpt2|dummy}.csv`, `results/04_em_weights_joint_{gpt2|dummy}.csv`  
- **`use_joint_em=True` only:** per-epoch worker-accuracy snapshots under `results/{joint|static}_tracking_{gpt2|dummy}/em_params_epoch_*.csv`. With **`use_joint_em=False`**, that directory is created but **no** per-epoch CSVs are written (static trust weights are fixed for the whole run).

**5. Animate EM parameter evolution**  

Run from `src/notebooks`. Prefer **`--path`** to the tracking folder that matches the run you want (the built-in `--mode standalone` / `--mode joint` shortcuts use folder and “true worker accuracy” paths that **do not** always match this repo — e.g. projected DPO uses `joint_tracking_gpt2/` / `joint_tracking_dummy/`, not plain `joint_tracking/`).

Ground-truth worker accuracies for the scatter plot must come from **`data/true_params/03_true_params.csv`** (written by **`simulate_crowd.py`**). Re-run simulation for the same annotator count as the tracking run before animating, or lengths may not match.

Examples:

```bash
cd src/notebooks
python ult_animate_clusters.py --path ../../results/standalone_tracking/
python ult_animate_clusters.py --path ../../results/joint_tracking_dummy/
python ult_animate_clusters.py --path ../../results/joint_tracking_gpt2/
python ult_animate_clusters.py --path ../../results/static_tracking_dummy/
```

Stress-test edge cases:

```bash
python ult_animate_clusters.py --path ../../results/edge_cases/isolated_sparsity_failure_tracking/
```

If the script reports a missing ground-truth file for `--path` edge runs, **`ult_animate_clusters.py` still looks under `data/processed/edge_cases/...`** for `03_true_params.csv`, but **`simulate_crowd.py` only writes** `data/true_params/03_true_params.csv` — align those paths in code or copy the CSV next to the edge-case data folder after that run.

---

## Phase 3 — Stress testing EM

**1. Parameter sweep**  
Sweeps sparsity `L`, adversary fraction, and annotator count `N`; runs simulation + EM per setting.

```bash
cd src/experiments
python stress_test.py
```

Outputs: `results/stress_tests/phase3_sweep_matrix.csv`, plus `data/processed/edge_cases/...` and `results/edge_cases/*_tracking/` for detected failure modes.

**Note:** the sweep repeatedly overwrites **`data/processed/02_train_noisy_votes.csv`** and **`data/true_params/03_true_params.csv`**. If you need a stable main dataset for Phases 1–2, re-run steps **2** and **3a** after stress testing, or archive those files before running `stress_test.py`.

**2. Heatmaps / line charts**

```bash
cd src/notebooks
python plot_stress_tests.py
```

Outputs: `results/stress_tests/heatmap_L_vs_Adv.png`, `results/stress_tests/linegraph_scale_N.png`

**3. Edge-case animations**  
After the sweep generates `*_tracking/` folders, use `ult_animate_clusters.py --path` as in Phase 2 step 5.
