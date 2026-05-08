Running instructions:
   Here we start with the demo...
# Baseline 
1. We load Anthropic/hh-rlhf, shuffle each official split with a fixed seed, then take up to 10000 train rows from `split="train"` and 2000 test rows from `split="test"` (fewer if the split is shorter). Running `ground_truth.py` as a script uses these defaults and a reproducible A/B label draw.

 - python src/data/ground_truth.py
 - results: data/processed/01_train_raw.csv, data/processed/01_test_raw.csv

 2. Generate the noisy dataset by simulating the crowd acting on this ground truth according to the variables of number of annotators, sparsity (number of tasks per worker), adverserial + spammer ratio. we generate the input for EM algorithm or naive DPO using the noisy morphed dataset, and store the true correctness parameters of each annotator.

 - python src/data/simulate_crowd.py
 - results: data/processed/02_train_noisy_params.csv, data/true_params/03_true_params.csv

 3. Tokenize the data for embeddings used by transformer and gpt2. 

 - python src/data/tokenize_data.py
 - results: data/tokenized/test_tokens.pt, data/tokenized/train_tokens.pt

 4. Now for this phase, we run a naive DPO trusting the annotators. This is not expected to be robust and produce good results without overfitting all the time. We have 2 options, a simple transformer and gpt2 (SFT done on a base model). 

 - python src/training/train_baseline_dpo.py
 - results: results/baseline_(sft or dummy)_metrics.csv, models/baseline_(gpt2 or dummy).pth - stores the weights of the learned model.

 5. We can now graph the train loss (dpo loss) and accuracy on ground truth with epochs to compare with other phases. 

 # EM introduction 

 1, 2, 3 are the same...

 4. Now we approach in 2 ways, EM on the noisy dataset to reaalise the correctness parameters and remove conflicts between workers for the same task, making dpo learn the true accpeted, rejected pairs, by making a csv on which dpo acts (standalone)   or with every epoch, taking the llm's predictions as the prior to the new update of parameters using EM, which makes it more confident in case of more conflicts. 

 - python src/models/em_standalone.py 
 - results: results/standalone_tracking/em_params_iter_{num}.csv, for num iterations, used for interactive plotting, results/04_em_weights.csv, results/05_em_inferred_params.csv, results/em_loss_history.csv

 - python src/training/train_projected_dpo.py 
 - results: standalone->joint in above, models/projected_dpo_model.pth, results/projected_metrics.csv

 5. For seeing the evolution of correctness parameters and estimated truth values of prompts, we draw an interactive graph.

 - python src/notebooks/ult_animate_cluters.py --mode (standalone or joint)
 - results: enjoy the graph...

 # Phase 3: Stress Testing the EM + DPO pathway
 1. The Parameter SweepWe loop through various combinations of Sparsity ($L$) and Adversary Ratios, running the EM algorithm on each configuration to find the isolated breaking points.
 
 - python src/experiments/stress_test.py
 - results: results/stress_tests/phase3_sweep_matrix.csv, and automatically generated isolated edge-case folders (e.g., results/edge_cases/isolated_sparsity_failure_tracking/).
 
 2. Visualize the Safe Zone plot the sweep matrix into a Heatmap, where crowdsourced RLHF remains viable.

- python src/notebooks/plot_stress_tests.py
- results: results/stress_tests/heatmap_L_vs_Adv.png,linegraph_scale_N.png

3. Animate the specific mathematical failures (e.g., watching the graph shatter when $L=1$, or watching a polarity flip when adversaries $>50\%$).

- python src/notebooks/ult_animate_clusters.py --path ../../results/edge_cases/[case_name]_tracking/
- results: Matplotlib animations depicting exactly how the system breaks under extreme pressure.

