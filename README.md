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

## Phase 1 — Data and baseline DPO

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

**3. Tokenize for GPT-2**

```bash
cd src/data
python tokenize_data.py
```

Outputs: `data/tokenized/train_tokens.pt`, `data/tokenized/test_tokens.pt`

**4. Baseline DPO (trusts raw votes)**  
Trains with **one loss term per vote row**: if the vote is B, A and B are swapped for that row’s DPO term (not majority-vote aggregation over workers). You can use **GPT-2** or a small **dummy** causal LM; edit `if __name__ == "__main__"` in `train_baseline_dpo.py` (`use_sft=True` for GPT-2, `False` for dummy; adjust `epochs` as needed).

```bash
cd src/training
python train_baseline_dpo.py
```

Outputs: `results/baseline_sft_metrics.csv` or `results/baseline_dummy_metrics.csv`, `models/baseline_gpt2.pth` or `models/baseline_dummy.pth`

**5. Plots**  
Use the CSV columns (`epoch`, `train_loss`, `test_accuracy`) in your notebook or tool of choice to compare loss and golden accuracy across epochs.

---

## Phase 2 — EM and projected DPO

Steps **1–3** are the same as above.

**4a. Standalone EM (offline weights)**  
Runs Dawid–Skene–style EM on `02_train_noisy_votes.csv` and writes trust weights for use by projected DPO when **not** using the joint loop.

```bash
cd src/models
python em_standalone.py
```

Outputs: `results/standalone_tracking/em_params_iter_*.csv`, `results/04_em_weights.csv`, `results/05_em_inferred_params.csv`, `results/em_loss_history.csv`

**4b. Projected DPO**  
Soft-label DPO using EM’s \(\gamma\) (trust weights). Two modes (see `if __name__ == "__main__"` in `train_projected_dpo.py`):

- **`use_joint_em=True`**: each epoch recomputes LLM priors from the current policy, runs EM (E-step + M-step), then trains with updated weights (no prior `04_em_weights.csv` required).
- **`use_joint_em=False`**: uses frozen **`results/04_em_weights.csv`** from **4a** — run `em_standalone.py` first.

```bash
cd src/training
python train_projected_dpo.py
```

Outputs: `results/joint_tracking/` (when joint EM is on), `models/projected_dpo_model.pth`, `results/projected_metrics.csv`

**5. Animate EM parameter evolution**

```bash
cd src/notebooks
python ult_animate_clusters.py --mode standalone
python ult_animate_clusters.py --mode joint
```

For stress-test edge cases, pass the tracking folder (paths relative to `src/notebooks`):

```bash
python ult_animate_clusters.py --path ../../results/edge_cases/isolated_sparsity_failure_tracking/
```

---

## Phase 3 — Stress testing EM

**1. Parameter sweep**  
Sweeps sparsity `L`, adversary fraction, and annotator count `N`; runs simulation + EM per setting.

```bash
cd src/experiments
python stress_test.py
```

Outputs: `results/stress_tests/phase3_sweep_matrix.csv`, plus `data/processed/edge_cases/...` and `results/edge_cases/*_tracking/` for detected failure modes.

**2. Heatmaps / line charts**

```bash
cd src/notebooks
python plot_stress_tests.py
```

Outputs: `results/stress_tests/heatmap_L_vs_Adv.png`, `results/stress_tests/linegraph_scale_N.png`

**3. Edge-case animations**  
After the sweep generates `*_tracking/` folders, use `ult_animate_clusters.py --path` as in Phase 2 step 5.
