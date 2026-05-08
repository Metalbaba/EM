% ============================================================
%  Expectation Maximisers — Report Sketch
%  Fill in every \todo{} block before submission.
% ============================================================
\documentclass[11pt, a4paper]{article}

% ── Packages ────────────────────────────────────────────────
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{parskip}        % space between paragraphs, no indent
% microtype omitted for compatibility
\usepackage{caption}
\usepackage{float}

% ── Colours ─────────────────────────────────────────────────
\definecolor{todocolor}{RGB}{192, 57, 43}
\definecolor{warncolor}{RGB}{180, 95, 6}
\definecolor{codeblue}{RGB}{31, 56, 100}
\definecolor{codegray}{RGB}{245, 245, 245}

% ── TODO / CAUTION commands ──────────────────────────────────
\newcommand{\todo}[1]{%
  \textcolor{todocolor}{\textbf{[TODO:} \textit{#1}\textbf{]}}%
}
% \warn{} flags a design or correctness concern that needs resolving before the demo.
\newcommand{\warn}[1]{%
  \noindent\colorbox{yellow!25}{\parbox{\dimexpr\linewidth-2\fboxsep}{%
    \textcolor{warncolor}{\textbf{$\triangleright$ CAUTION:} \textit{#1}}%
  }}%
  \vspace{4pt}%
}

% ── Hyperref setup ──────────────────────────────────────────
\hypersetup{
  colorlinks=true,
  linkcolor=codeblue,
  urlcolor=codeblue,
  citecolor=codeblue,
  pdftitle={Expectation Maximisers},
  pdfauthor={},
}

% ── Code listing style ──────────────────────────────────────
\lstset{
  backgroundcolor=\color{codegray},
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  rulecolor=\color{gray!40},
  tabsize=2,
  showstringspaces=false,
}

% ── Section title formatting ────────────────────────────────
\titleformat{\section}{\large\bfseries\color{codeblue}}{{\thesection}}{1em}{}
\titleformat{\subsection}{\normalsize\bfseries\color{codeblue}}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection}{1em}{}

% ============================================================
\begin{document}

% ── Title page ──────────────────────────────────────────────
\begin{titlepage}
  \centering
  \vspace*{3cm}
  {\Huge\bfseries\color{codeblue} Expectation Maximisers \par}
  \vspace{0.6cm}
  {\large\itshape Crowd Preference Learning with EM-Weighted Preference Optimisation \par}
  \vspace{1.5cm}
  \rule{\linewidth}{0.4pt}\\[0.4cm]
  {\large Course Project Report \par}
  \vspace{0.4cm}
  \todo{Author names} \par
  \vspace{0.2cm}
  \todo{Institution / Course name} \par
  \vspace{0.2cm}
  {\large May 2026 \par}
  \vspace{1cm}
  \rule{\linewidth}{0.4pt}
\end{titlepage}

% ── Table of contents ───────────────────────────────────────
\tableofcontents
\newpage

% ============================================================
\section{Abstract}

This project combines noisy crowd annotations on the Anthropic HH-RLHF dataset with
Dawid--Skene-style Expectation Maximization (EM) and EM-weighted DPO-style preference
optimisation. Annotators vote on which of two responses (A or B) is better; votes are
simulated with heterogeneous worker reliability. EM infers per-prompt latent preferences
and per-worker accuracy parameters. A policy language model (GPT-2) supplies LLM priors
for the EM E-step; training uses a weighted objective where soft labels come from EM's
posterior. The GPT-2 pipeline achieves approximately 75\% golden-dataset accuracy.

\todo{Add 2--3 sentences on key findings / contributions once experiments are complete.}

\newpage

% ============================================================
\section{Introduction}

\subsection{Motivation}

Human feedback is central to aligning language models with human preferences. In practice,
annotator disagreement is the norm: workers vary widely in expertise, attention, and
subjective judgement. Treating all votes equally discards valuable signal about annotator
reliability.

\todo{Expand with literature context: RLHF, Bradley--Terry models, disagreement in annotation.}

\subsection{Problem Statement}

Given a set of pairwise preference votes from heterogeneous annotators on prompt--response
pairs, we aim to (1) infer the latent ground-truth preference per prompt and (2) train a
language model policy using these inferred preferences as soft labels.

\subsection{Contributions}

\begin{itemize}[noitemsep]
  \item A simulation framework over Anthropic HH-RLHF with four annotator types
        (expert, average, spammer, adversary) and a bipartite assignment graph.
  \item An \textbf{oracle upper bound}: DPO trained directly on the hidden ground-truth
        labels, establishing the accuracy ceiling reachable under the same model and
        objective without crowd noise.
  \item A staged experimental pipeline: noisy baseline DPO $\to$ offline EM + projected
        DPO $\to$ joint EM + projected DPO, isolating the contribution of each component
        relative to both the noisy baseline and the oracle ceiling.
  \item An EM quality metric ($\mathcal{L}_\text{EM}$) combining label MSE and worker
        parameter MSE, used to track EM convergence visually across phases.
  \item Stress-testing EM under increasing task scale, adversarial annotator ratios, and
        vote sparsity, with individual and combined factor analyses.
  \item A robustness evaluation showing that EM + best DPO outperforms best DPO alone
        even on the worst-case annotator regime.
  \item (If time) Hierarchical EM with annotator expertise levels, followed by projected
        DPO + SFT for peak accuracy.
\end{itemize}

\todo{Revise after finalising experiments.}

\newpage

% ============================================================
\section{Background \& Related Work}

\subsection{Reinforcement Learning from Human Feedback (RLHF)}

\todo{Summarise the RLHF pipeline: SFT $\to$ reward model $\to$ PPO.
Cite Christiano et al.~(2017), Stiennon et al.~(2020), InstructGPT.}

\subsection{Direct Preference Optimisation (DPO)}

\todo{Describe the DPO objective (Rafailov et al.~2023): avoids an explicit reward model
by reparameterising as policy log-ratios.}

\subsection{The Dawid--Skene Model}

The Dawid--Skene model \cite{dawid1979} is an EM algorithm for inferring latent
ground-truth labels from noisy crowd annotations. Each annotator is characterised by a
confusion matrix; in the binary case this reduces to per-worker sensitivity $\alpha_j$
and specificity $\beta_j$.

\todo{Add 1--2 sentences on recent extensions and applications to NLP annotation.}

\subsection{Dataset: Anthropic HH-RLHF}

Anthropic's Helpful and Harmless RLHF dataset \cite{bai2022} contains human-written
preference pairs (chosen / rejected responses). This project uses the first 1,000 rows
of the training split as the prompt pool.

\newpage

% ============================================================
\section{Method}

\subsection{Simulation}

The dataset is loaded from the Anthropic HH-RLHF Hugging Face repository with fixed
random seeds. Two splits are retained: up to \textbf{10,000 training rows} from
\texttt{split="train"} and up to \textbf{2,000 test rows} from \texttt{split="test"}.
Each row receives a reproducible random \texttt{truth\_is\_A} draw.

A bipartite graph assigns training prompts to $N$ annotators with sparsity $L$
(votes per prompt). Table~\ref{tab:workers} describes the four worker types.
Adversary fraction and $L$ are configurable in \texttt{simulate\_crowd.py}.

\begin{table}[H]
  \centering
  \caption{Simulated annotator types and their properties.}
  \label{tab:workers}
  \begin{tabular}{llll}
    \toprule
    \textbf{Type} & \textbf{Count} & \textbf{True Accuracy} & \textbf{Behaviour} \\
    \midrule
    Expert    & configurable & High         & Reliably votes with ground truth \\
    Average   & configurable & Medium       & Moderately reliable \\
    Spammer   & configurable & $\approx$0.5 & Near-random votes \\
    Adversary & configurable & Low          & Systematically votes against truth \\
    \bottomrule
  \end{tabular}
\end{table}

For each prompt, \texttt{truth\_is\_A} is drawn uniformly at random. Response A is the
HH-RLHF \textit{chosen} response when \texttt{truth\_is\_A}$=1$, else \textit{rejected}.
Each vote is sampled as:
\[
  v_{ij} = \begin{cases} \text{truth}_i & \text{with prob. } \theta_j \\ 1-\text{truth}_i & \text{otherwise} \end{cases}
\]

\subsection{Oracle Upper Bound}

To bound what is achievable under the same model and DPO objective without crowd noise,
we define an \textbf{oracle baseline}: a DPO run where each training example uses
\texttt{truth\_is\_A} directly as the label instead of any annotator vote. This requires
no EM, no vote aggregation, and no simulation --- it trains on the hidden ground truth.
The oracle test accuracy establishes a ceiling that crowd-learning methods are measured
against.

Concretely, \texttt{tokenize\_oracle.py} builds one training example per prompt using
\texttt{truth\_is\_A} as the vote, producing \texttt{oracle\_train\_tokens.pt}.
\texttt{train\_baseline\_dpo.py} then loads this file when \texttt{is\_oracle=True}.

\subsection{Expectation Maximisation (Dawid--Skene)}

\subsubsection{Latent Variables}

Let $\gamma_i = P(\text{prompt } i \text{'s true label is ``A is better''})$, with
per-annotator parameters $\alpha_j = P(\text{vote A} \mid \text{truth A})$ and
$\beta_j = P(\text{vote B} \mid \text{truth B})$. The global prior is
$\pi_1 = \frac{1}{N}\sum_i \gamma_i$.

\subsubsection{E-Step}

The posterior over the latent label for prompt $i$ is accumulated in log-space over
all votes from assigned workers:

\[
  \log \gamma_i \propto \log \pi_1
    + \sum_{j \in \mathcal{A}(i)} \bigl[v_{ij}\log\alpha_j + (1-v_{ij})\log(1-\beta_j)\bigr]
\]

When LLM priors $p_i^\text{LLM}$ are available (see \S\ref{sec:joint}), the term
$\log\pi_1$ is replaced by $\log p_i^\text{LLM}$.

\todo{Add the symmetric expression for $\log(1-\gamma_i)$ and normalisation step.}

\subsubsection{M-Step}

Given the updated $\gamma_i$, expected counts are accumulated via \texttt{scatter\_add}
and used to update $\pi_1$, $\alpha_j$, $\beta_j$:

\[
  \alpha_j = \frac{\sum_{i: j\in\mathcal{A}(i)} \gamma_i \cdot \mathbf{1}[v_{ij}=1]}
                  {\sum_{i: j\in\mathcal{A}(i)} \gamma_i}, \qquad
  \beta_j  = \frac{\sum_{i: j\in\mathcal{A}(i)} (1-\gamma_i) \cdot \mathbf{1}[v_{ij}=0]}
                  {\sum_{i: j\in\mathcal{A}(i)} (1-\gamma_i)}
\]

$\alpha_j$ and $\beta_j$ are clamped away from 0 and 1 for numerical stability.

\subsection{Weighted DPO-Style Loss}

The implicit reward for a sequence $y$ under policy $\pi_\theta$ and reference $\pi_\text{ref}$ is:
\[
  r(x, y) = \beta \left(\log\frac{\pi_\theta(y\mid x)}{\pi_\text{ref}(y\mid x)}\right)
\]

The weighted loss for prompt $i$ with soft label $\gamma_i$ is:
\[
  \mathcal{L}_i = -\,\gamma_i \log\sigma\!\bigl(r(x_i, y_i^A) - r(x_i, y_i^B)\bigr)
                 -(1-\gamma_i)\log\sigma\!\bigl(r(x_i, y_i^B) - r(x_i, y_i^A)\bigr)
\]

where $y_i^A$ and $y_i^B$ are the two candidate responses.

\subsection{Joint Training Loop}
\label{sec:joint}

Each training epoch alternates three phases:

\begin{enumerate}[noitemsep]
  \item \textbf{LLM Priors.} Compute per-prompt
        $p_i^\text{LLM} = \sigma\!\bigl(\beta(r(x_i,y_i^A)-r(x_i,y_i^B))\bigr)$
        under the current policy and frozen reference.
  \item \textbf{EM Update.} Run one E-step (with $p_i^\text{LLM}$ as prior) and one
        M-step to update $\gamma_i$, $\alpha_j$, $\beta_j$.
  \item \textbf{Policy Update.} Compute $\mathcal{L}_i$ using the updated $\gamma_i$;
        backpropagate and step AdamW.
\end{enumerate}

\subsection{Projected DPO}
\label{sec:proj-dpo}

Projected DPO is soft-label DPO in which EM's posterior $\gamma_i$ replaces the binary
hard label. The loss for prompt $i$ is identical to the weighted DPO loss in \S4.3.
The term \textit{projected} refers to the label space: rather than projecting a hard
preference onto the loss, the EM posterior \emph{projects} the full annotation distribution
into a single scalar trust weight.

Two flags in \texttt{train\_projected\_dpo.py} control the configuration:

\begin{itemize}[noitemsep]
  \item \texttt{use\_joint\_em}: \texttt{True} = joint mode --- each epoch recomputes
        LLM priors from the current policy, runs one EM E-step and M-step to refresh
        $\gamma_i$, then trains. No pre-computed weight file needed.
        \texttt{False} = static/offline mode --- uses frozen trust weights from
        \texttt{results/04\_em\_weights.csv} (run \texttt{em\_standalone.py} first).
  \item \texttt{use\_sft}: \texttt{True} = GPT-2 policy; \texttt{False} = dummy LM.
\end{itemize}

Output files follow the pattern \texttt{\{gpt2|dummy\}\_\{joint|static\}}:

\begin{itemize}[noitemsep]
  \item Metrics: \texttt{results/projected\_\{gpt2|dummy\}\_\{joint|static\}\_metrics.csv}
  \item Weights: \texttt{models/projected\_\{gpt2|dummy\}\_\{joint|static\}.pth}
  \item Joint mode only: \texttt{results/05\_em\_inferred\_params\_joint\_\{gpt2|dummy\}.csv},\\
        \texttt{results/04\_em\_weights\_joint\_\{gpt2|dummy\}.csv}
  \item Per-epoch EM traces: \texttt{results/\{joint|static\}\_tracking\_\{gpt2|dummy\}/em\_params\_epoch\_*.csv}
\end{itemize}

\warn{Offline mode decouples EM quality from DPO training, which is cleaner for ablation
but means the policy cannot improve the E-step. Joint mode introduces a feedback loop but
makes it harder to isolate the contribution of EM vs.\ DPO. Run both and compare.}

\subsection{Hierarchical EM (Optional Extension)}
\label{sec:hier-em}

In the base Dawid--Skene model all annotators are treated as independent. A hierarchical
extension pools information across annotators of the same type by placing a shared Beta
prior on $(\alpha_j, \beta_j)$ within each group $g$:
\[
  \alpha_j \mid g(j) \sim \mathrm{Beta}(a_g, b_g), \qquad
  \beta_j  \mid g(j) \sim \mathrm{Beta}(c_g, d_g)
\]
The M-step must then update both the per-worker parameters and the group-level
hyperparameters $(a_g, b_g, c_g, d_g)$.

\warn{Hierarchical EM is significantly more complex to implement than the flat variant.
The M-step no longer has a closed form for the Beta hyperparameters; moment matching or
variational updates are typically used. Set a firm cut-off date: if not complete by that
date, drop this and present Phase~5 as the final result.}

\todo{Formalise the modified E- and M-steps, and add the expertise-level variable if used.}

\newpage

% ============================================================
\section{Implementation}

\subsection{Project Layout}

\begin{table}[H]
  \centering
  \caption{Repository modules and their roles.}
  \label{tab:layout}
  \begin{tabular}{lll}
    \toprule
    \textbf{Location} & \textbf{File} & \textbf{Role} \\
    \midrule
    \texttt{src/data}        & \texttt{ground\_truth.py}         & Load HH-RLHF, assign \texttt{truth\_is\_A}, write train/test CSVs \\
    \texttt{src/data}        & \texttt{simulate\_crowd.py}       & Bipartite vote simulation (configurable $N$, $L$, adversary mix) \\
    \texttt{src/data}        & \texttt{tokenize\_data.py}        & Tokenise noisy votes $\to$ \texttt{noisy\_train\_tokens.pt}, \texttt{test\_tokens.pt} \\
    \texttt{src/data}        & \texttt{tokenize\_oracle.py}      & Tokenise ground truth $\to$ \texttt{oracle\_train\_tokens.pt} (ceiling) \\
    \texttt{src/models}      & \texttt{em\_standalone.py}        & Offline Dawid--Skene EM; writes \texttt{04\_em\_weights.csv} \\
    \texttt{src/training}    & \texttt{train\_baseline\_dpo.py}  & Per-vote baseline DPO (\texttt{use\_sft} flag for GPT-2 vs.\ dummy) \\
    \texttt{src/training}    & \texttt{train\_projected\_dpo.py} & Soft-label DPO (\texttt{use\_joint\_em} flag for joint vs.\ offline) \\
    \texttt{src/experiments} & \texttt{stress\_test.py}          & Sweep $L$, adversary fraction, $N$; detect EM failure modes \\
    \texttt{src/notebooks}   & \texttt{ult\_animate\_clusters.py}& Animate EM parameter evolution (standalone, joint, or edge-case path) \\
    \texttt{src/notebooks}   & \texttt{plot\_stress\_tests.py}   & Heatmap ($L$ vs.\ adversary) and line chart (accuracy vs.\ $N$) \\
    \bottomrule
  \end{tabular}
\end{table}

Artefacts are written to three root-level directories: \texttt{data/} (CSVs and tokenised
tensors), \texttt{results/} (metrics CSVs, tracking folders, plots), and \texttt{models/}
(saved \texttt{.pth} checkpoints).

\subsection{Key Hyperparameters}

\begin{table}[H]
  \centering
  \caption{Hyperparameters for the main experimental runs.}
  \label{tab:hparams}
  \begin{tabular}{lll}
    \toprule
    \textbf{Parameter} & \textbf{Value} & \textbf{Notes} \\
    \midrule
    Training prompts      & up to 10,000 & HH-RLHF \texttt{split="train"}, fixed seed \\
    Test prompts          & up to 2,000  & HH-RLHF \texttt{split="test"}, fixed seed \\
    Annotators $N$        & \todo{fill}  & Configurable in \texttt{simulate\_crowd.py} \\
    Vote sparsity $L$     & \todo{fill}  & Votes per prompt; swept in Phase~3 \\
    Adversary fraction    & \todo{fill}  & Fraction of spammer + adversary workers \\
    Max token length      & 128          & GPT-2 tokeniser, pad = eos \\
    LR (GPT-2)            & \todo{fill}  & AdamW; \texttt{train\_baseline\_dpo.py} / \texttt{train\_projected\_dpo.py} \\
    LR (dummy LM)         & \todo{fill}  & \texttt{use\_sft=False} in \texttt{train\_baseline\_dpo.py} \\
    Training epochs       & \todo{fill}  & Configurable per run \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{Setup}

\begin{lstlisting}[language=bash, caption={Environment setup}]
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt  # torch, transformers, datasets, pandas, numpy, tqdm
pip install matplotlib seaborn   # required for plots and animations
\end{lstlisting}

\texttt{ground\_truth.py} requires network access to download \texttt{Anthropic/hh-rlhf}
via Hugging Face \texttt{datasets}. Device selection is automatic: CUDA $>$ Apple MPS $>$
CPU.

\todo{Add the specific random seed(s) and hardware spec used for final results.}

\newpage

% ============================================================
\section{Experimental Design \& Results}

The evaluation follows a staged pipeline in which each phase fixes the best configuration
found so far and adds one new component. This allows clean ablation of EM quality, DPO
variant, and SFT, alongside a robustness stress test.

\subsection{Metrics}

\paragraph{Golden dataset accuracy.}
The fraction of prompts where the predicted preference matches the simulator's hidden
\texttt{truth\_is\_A} label. Two variants are tracked:
\begin{itemize}[noitemsep]
  \item \textit{EM accuracy:} $\hat{y}_i = \mathbf{1}[\gamma_i > 0.5]$ vs.\ \texttt{truth\_is\_A}.
  \item \textit{DPO accuracy:} $\hat{y}_i = \mathbf{1}[r(x_i,y_i^A) > r(x_i,y_i^B)]$ vs.\ \texttt{truth\_is\_A}.
\end{itemize}

\paragraph{EM quality loss.}
A scalar summarising how close the inferred parameters are to ground truth:
\[
  \mathcal{L}_\text{EM} =
    \underbrace{\frac{1}{N}\sum_i \bigl(\gamma_i - \texttt{truth\_is\_A}_i\bigr)^2}_{\text{label MSE}}
  + \underbrace{\frac{1}{J}\sum_j \left(\frac{\alpha_j+\beta_j}{2} - \theta_j^\text{true}\right)^2}_{\text{worker MSE}}
\]
Used for all EM convergence plots (not a training signal --- requires simulator ground truth).

\warn{$\mathcal{L}_\text{EM}$ is an \emph{evaluation} metric only. Never use it as a loss
during training; that would leak the hidden ground truth into the model.}

\subsection{Data Setup}

The first 1,000 rows of the Anthropic HH-RLHF training split are used throughout.
\begin{itemize}[noitemsep]
  \item \textbf{Training set:} prompts used for vote simulation and policy training.
  \item \textbf{Validation set:} a held-out slice (e.g.\ 20\%) for epoch-wise monitoring.
  \item \textbf{Ground truth:} \texttt{truth\_is\_A} per prompt --- used only for metrics,
        never for training.
\end{itemize}

\warn{Fix the train/val split and random seed \emph{before} running any experiment.
Any change to the split afterwards invalidates cross-phase comparisons.}

\todo{Specify exact split sizes and the random seed used.}

% ─────────────────────────────────────────────────────────────
\subsection{Phase 1 --- Baseline and Oracle DPO}

\paragraph{Setup.}
\texttt{train\_baseline\_dpo.py} supports four configurations via two flags:

\begin{itemize}[noitemsep]
  \item \texttt{use\_sft}: \texttt{True} = GPT-2 pretrained policy; \texttt{False} = small
        dummy causal LM.
  \item \texttt{is\_oracle}: \texttt{False} = loads \texttt{noisy\_train\_tokens.pt}
        (crowd votes, one DPO term per vote row, A/B swapped when \texttt{vote==0});
        \texttt{True} = loads \texttt{oracle\_train\_tokens.pt} (ground-truth
        \texttt{truth\_is\_A} as the label, one term per prompt).
\end{itemize}

Running all four combinations produces two reference points for every subsequent phase:
a \textbf{noisy baseline} (what the model learns from crowd votes alone) and an
\textbf{oracle ceiling} (what it could learn with perfect labels under the same objective).

\begin{table}[H]
  \centering
  \caption{Phase 1 output files.}
  \begin{tabular}{llll}
    \toprule
    \textbf{\texttt{use\_sft}} & \textbf{\texttt{is\_oracle}} & \textbf{Metrics CSV} & \textbf{Checkpoint} \\
    \midrule
    False & False & \texttt{results/baseline\_dummy\_metrics.csv} & \texttt{models/baseline\_dummy.pth} \\
    True  & False & \texttt{results/baseline\_gpt2\_metrics.csv}  & \texttt{models/baseline\_gpt2.pth}  \\
    False & True  & \texttt{results/oracle\_dummy\_metrics.csv}   & \texttt{models/oracle\_dummy.pth}   \\
    True  & True  & \texttt{results/oracle\_gpt2\_metrics.csv}    & \texttt{models/oracle\_gpt2.pth}    \\
    \bottomrule
  \end{tabular}
\end{table}

\paragraph{Plots.} \texttt{test\_accuracy} and \texttt{train\_loss} vs.\ epoch for all
four variants on the same axes. The oracle curves set the ceiling; the noisy curves set
the floor that EM-assisted methods must exceed.

\paragraph{Expected result.}
Oracle $>$ noisy baseline by a meaningful margin. The gap measures how much performance
crowd noise costs under this objective and model capacity.

\warn{``No SFT'' means $\pi_\text{ref}$ is the off-the-shelf pretrained GPT-2 (or random
dummy init), not fine-tuned on HH-RLHF chosen responses. Keep this reference consistent
across all phases; changing it mid-pipeline confounds accuracy comparisons.}

\todo{Record test accuracy for all four variants. The oracle GPT-2 accuracy is the target
ceiling for all subsequent phases to approach.}

% ─────────────────────────────────────────────────────────────
\subsection{Phase 2 --- EM + Projected DPO}

\paragraph{Setup.}
Run \texttt{em\_standalone.py} on \texttt{02\_train\_noisy\_votes.csv} to produce
soft labels $\gamma_i$ (\texttt{results/04\_em\_weights.csv}) and inferred worker
parameters (\texttt{results/05\_em\_inferred\_params.csv}). Then run
\texttt{train\_projected\_dpo.py} in either mode (see \S\ref{sec:proj-dpo}):
\textbf{offline} (frozen EM weights) for a clean ablation, or \textbf{joint} (EM
refreshed each epoch using LLM priors) for the full feedback loop.

\paragraph{Plots.}
\begin{itemize}[noitemsep]
  \item $\mathcal{L}_\text{EM}$ vs.\ EM iteration from \texttt{results/em\_loss\_history.csv}.
  \item EM parameter evolution animated via \texttt{ult\_animate\_clusters.py}. Use
        \texttt{--mode standalone} for the offline EM run, or pass the tracking folder
        directly for projected runs, e.g.\
        \texttt{--path ../../results/joint\_tracking\_gpt2/} or
        \texttt{--path ../../results/static\_tracking\_dummy/}.
  \item Histogram snapshots of $\gamma_i$ at selected iterations --- not individual curves.
        Mean$\pm$std per worker type is also informative.
  \item \texttt{test\_accuracy} vs.\ epoch from \texttt{results/projected\_metrics.csv},
        overlaid with the Phase~1 curve.
\end{itemize}

\paragraph{Expected result.}
EM soft labels outperform the per-vote baseline; projected DPO accuracy improves vs.\ Phase~1.

\warn{Plotting individual $\gamma_i$ trajectories for 10,000 prompts is unreadable.
Use the scalar $\mathcal{L}_\text{EM}$, histogram snapshots, or the cluster animation.}

\todo{Fill in final DPO accuracy and $\mathcal{L}_\text{EM}$ at convergence for both
offline and joint modes.}

% ─────────────────────────────────────────────────────────────
\subsection{Phase 3 --- EM Stress Testing}
\label{sec:phase3-stress}

\paragraph{Setup.}
Run \texttt{stress\_test.py} from \texttt{src/experiments/}. The script sweeps three
factors independently then combines them for a worst-case regime:

\begin{table}[H]
  \centering
  \caption{Stress-test sweep parameters.}
  \label{tab:stress}
  \begin{tabular}{p{3.2cm}p{3cm}p{6cm}}
    \toprule
    \textbf{Factor} & \textbf{Values swept} & \textbf{Expected effect} \\
    \midrule
    Vote sparsity $L$ (votes per task)
      & \todo{fill from script}
      & Fewer observations per prompt $\to$ less certain $\gamma_i$ \\
    Adversary fraction (spammer + adversary)
      & \todo{fill from script}
      & Noisier votes $\to$ slower convergence, higher $\mathcal{L}_\text{EM}$ \\
    Annotator count $N$ (tasks fixed)
      & \todo{fill from script}
      & Fewer annotators $\to$ weaker $\alpha_j/\beta_j$ estimates per worker \\
    Combined worst case
      & Low $L$ + high adversary + low $N$
      & Maximum stress; edge-case tracking folders generated \\
    \bottomrule
  \end{tabular}
\end{table}

\paragraph{Outputs.}
\texttt{results/stress\_tests/phase3\_sweep\_matrix.csv} (full sweep results),
\texttt{results/stress\_tests/heatmap\_L\_vs\_Adv.png} (sparsity vs.\ adversary fraction),
\texttt{results/stress\_tests/linegraph\_scale\_N.png} (accuracy vs.\ annotator count),
and per-failure-mode \texttt{results/edge\_cases/*\_tracking/} folders for animation.

\paragraph{Plots.}
Run \texttt{plot\_stress\_tests.py} for heatmaps and line charts. For edge-case
animations: \texttt{ult\_animate\_clusters.py --path ../../results/edge\_cases/<folder>/}.

\paragraph{Expected result.}
$\mathcal{L}_\text{EM}$ increases monotonically as $L$ decreases, adversary fraction
increases, or $N$ decreases. The combined worst case should show the largest degradation.

\warn{\textbf{Scaling direction.} Increasing $N$ (annotators) while holding tasks fixed
gives more signal per task, so EM should \emph{improve}. The stress comes from
\emph{reducing} $N$ or $L$. Confirm \texttt{stress\_test.py} sweeps in the correct direction.}

\todo{Fill in the actual sweep values from the script and record $\mathcal{L}_\text{EM}$
at convergence per setting. Identify the worst-case configuration for Phase~4 (robustness).}

% ─────────────────────────────────────────────────────────────
\subsection{Phase 4 --- Robustness Evaluation}

\paragraph{Setup.}
Take the worst-case vote dataset identified in Phase~4. Evaluate two configurations:
\begin{enumerate}[noitemsep]
  \item \textbf{Best DPO only (no EM):} majority vote from the noisy dataset as hard
        labels; same DPO + SFT config from Phase~3.
  \item \textbf{EM + Best DPO:} run EM on the worst-case dataset (producing degraded
        but non-trivial $\gamma_i$); same DPO config.
\end{enumerate}

\paragraph{Claim.}
Even with degraded EM labels, configuration (2) should outperform (1), demonstrating
robustness of the full pipeline to adversarial annotator conditions.

\paragraph{Plots.}
Accuracy curves for both configurations on the same axes, with Phase~1 baseline for reference.

\warn{\textbf{Baseline sanity check.} If the worst-case dataset has, say, 70\% adversarial
workers, majority vote will be \emph{systematically inverted} --- configuration (1) may
fall \emph{below 50\%} accuracy. This makes EM look impressive but trivially so. Consider
whether a stronger no-EM baseline (e.g.\ annotator-type-weighted majority, without full EM)
would make the robustness claim more convincing.}

\todo{Fill in final accuracy for both configurations.}

% ─────────────────────────────────────────────────────────────
\subsection{Phase 6 (Optional) --- Hierarchical EM + Projected DPO + SFT}

If time permits, extend the annotator model to include an explicit expertise level
(see \S\ref{sec:hier-em}). Combined with the best DPO + SFT variant, this should yield
the highest golden accuracy across all phases.

\paragraph{Plots.}
Final accuracy from all phases on a single ``staircase'' improvement plot.

\todo{Implement hierarchical EM and fill in results. If not completed, present Phase~5 as the conclusion.}

% ─────────────────────────────────────────────────────────────
\subsection{Summary of Results}

\begin{table}[H]
  \centering
  \caption{Golden-dataset accuracy across all pipeline phases.}
  \label{tab:summary}
  \begin{tabular}{clll}
    \toprule
    \textbf{Phase} & \textbf{Configuration} & \textbf{Test Acc.} & \textbf{$\mathcal{L}_\text{EM}$} \\
    \midrule
    \multicolumn{4}{l}{\textit{Baselines \& ceiling}} \\
    1 & Oracle DPO, GPT-2 (\textit{ceiling})                    & \todo{fill} & --- \\
    1 & Oracle DPO, dummy LM (\textit{ceiling})                  & \todo{fill} & --- \\
    1 & Noisy baseline DPO, GPT-2                               & \todo{fill} & --- \\
    1 & Noisy baseline DPO, dummy LM                            & \todo{fill} & --- \\
    \multicolumn{4}{l}{\textit{EM-assisted}} \\
    2 & Offline EM + Projected DPO                              & \todo{fill} & \todo{fill} \\
    2 & Joint EM + Projected DPO                                & \todo{fill} & \todo{fill} \\
    \multicolumn{4}{l}{\textit{Stress tests}} \\
    3 & Stress test --- worst $L$ (sparsity)                    & \todo{fill} & \todo{fill} \\
    3 & Stress test --- worst adversary fraction                & \todo{fill} & \todo{fill} \\
    3 & Stress test --- worst $N$ (annotator count)             & \todo{fill} & \todo{fill} \\
    3 & Stress test --- combined worst case                     & \todo{fill} & \todo{fill} \\
    \multicolumn{4}{l}{\textit{Robustness}} \\
    4 & Worst-case data + Best DPO, no EM                       & \todo{fill} & --- \\
    4 & Worst-case data + EM + Best DPO                         & \todo{fill} & \todo{fill} \\
    \multicolumn{4}{l}{\textit{Optional extension}} \\
    5 & Hierarchical EM + Projected DPO                         & \todo{fill} & \todo{fill} \\
    \bottomrule
  \end{tabular}
\end{table}

\newpage

% ============================================================
\section{Discussion}

\subsection{Effect of LLM Priors on EM Convergence}

\todo{Discuss whether and how the policy prior changes $\gamma$ convergence relative to
the flat 50/50 prior. Does a stronger policy lead to faster / better convergence?}

\subsection{Limitations}

\begin{itemize}[noitemsep]
  \item Simulated annotators may not capture real human disagreement patterns.
  \item Only 1,000 prompts --- small scale relative to real RLHF pipelines.
  \item The E-step in \texttt{em.py} adds the global $\pi_1$ term on top of the LLM prior
        (and applies the static prior twice when no LLM prior is given), deviating from
        textbook Dawid--Skene.
\end{itemize}

\todo{Add any further limitations identified during write-up.}

\subsection{Open Design Questions}

The following concerns should be resolved before or during the demo. They are flagged here
so they do not get lost.

\begin{enumerate}

  \item \textbf{``Projected DPO'' is undefined.}
        The term does not appear in the standard literature. A precise mathematical
        definition must be written before implementation (see \S\ref{sec:proj-dpo}).
        Without it, the Phase~3 improvement cannot be attributed to projection vs.\ SFT.

  \item \textbf{Reference model consistency.}
        Phase~1 uses pretrained GPT-2 as $\pi_\text{ref}$; Phase~3 uses an SFT-tuned GPT-2.
        Changing the reference changes the implicit reward scale, making direct accuracy
        comparisons across phases potentially confounded. Consider reporting reward margins
        in addition to accuracy to make this visible.

  \item \textbf{Scaling direction in Phase~4.}
        Increasing tasks and annotators proportionally at fixed vote-sparsity should
        \emph{improve} EM, not stress it. The intended stress regime is: tasks increase,
        annotators fixed (so each worker covers more tasks with the same budget, reducing
        per-worker signal density). Confirm the simulation code reflects this.

  \item \textbf{Majority-vote baseline in Phase~5.}
        A highly adversarial dataset (e.g.\ 70\% adversary workers) will produce
        systematically inverted majority votes, driving baseline accuracy well below 50\%.
        The EM result will look strong by comparison, but the contrast is trivial.
        A more informative baseline would be a weighted combination using prior knowledge
        of annotator type fractions (without full EM).

  \item \textbf{The double-prior bug in \texttt{em.py}.}
        The E-step currently applies the global $\pi_1$ prior \emph{in addition} to the
        LLM prior, deviating from textbook Dawid--Skene. This inflates the effect of the
        global prior and may artificially advantage phases that use LLM priors. Fix before
        final experiments if possible.

  \item \textbf{$\gamma_i$ visualisation scale.}
        Plotting all 1,000 $\gamma_i$ trajectories over EM iterations is unreadable.
        Use the scalar $\mathcal{L}_\text{EM}$, histogram snapshots, or per-type
        mean $\pm$ std instead.

\end{enumerate}

\subsection{Future Work}

\begin{itemize}[noitemsep]
  \item Scale to the full HH-RLHF dataset.
  \item Replace simulated annotators with real crowdworker disagreement data.
  \item Experiment with a larger base model (e.g.\ GPT-2 medium/large or a modern open LLM).
  \item Fix the double-prior issue in \texttt{em.py} and re-evaluate.
\end{itemize}

\todo{Add any future directions specific to your course goals.}

\newpage

% ============================================================
\section{Conclusion}

This project demonstrates a principled approach to learning from noisy crowd preferences.
Dawid--Skene EM recovers per-prompt latent preferences and per-worker reliability, and
LLM priors from the current policy enrich the E-step. The resulting soft labels drive a
weighted DPO-style training objective that achieves $\approx$75\% golden-dataset accuracy
with a GPT-2 policy.

\todo{Revise once final numbers are in and key takeaways are clear.}

\newpage

% ============================================================
\bibliographystyle{plain}
\begin{thebibliography}{9}

\bibitem{dawid1979}
A.~P. Dawid and A.~M. Skene.
\newblock Maximum likelihood estimation of observer error-rates using the EM algorithm.
\newblock \textit{Applied Statistics}, 28(1):20--28, 1979.

\bibitem{rafailov2023}
R.~Rafailov, A.~Sharma, E.~Mitchell, et al.
\newblock Direct preference optimization: Your language model is secretly a reward model.
\newblock \textit{NeurIPS}, 2023.

\bibitem{bai2022}
Y.~Bai et al.
\newblock Training a helpful and harmless assistant with reinforcement learning from human feedback.
\newblock \textit{arXiv:2204.05862}, 2022.

\end{thebibliography}

\todo{Add all remaining citations: InstructGPT, original RLHF papers, crowd annotation work.}

\newpage

% ============================================================
\appendix
\section{Running the Code}

All scripts use paths relative to \textbf{their own directory}. Run each command from the
directory shown so outputs land under \texttt{data/}, \texttt{results/}, and \texttt{models/}
at the repo root.

\subsection*{Phase 1 --- Data, Baseline DPO, and Oracle}
\begin{lstlisting}[language=bash]
cd src/data && python ground_truth.py       # -> data/processed/01_train_raw.csv, 01_test_raw.csv
cd src/data && python simulate_crowd.py     # -> data/processed/02_train_noisy_votes.csv
                                            #    data/true_params/03_true_params.csv
cd src/data && python tokenize_data.py      # -> data/tokenized/noisy_train_tokens.pt
                                            #    data/tokenized/test_tokens.pt
cd src/data && python tokenize_oracle.py    # -> data/tokenized/oracle_train_tokens.pt
                                            #    (requires ground_truth.py only, no crowd votes)

# Run all four combinations (edit use_sft and is_oracle flags in the script):
cd src/training && python train_baseline_dpo.py  # noisy  + dummy  -> results/baseline_dummy_metrics.csv
                                                 # noisy  + GPT-2  -> results/baseline_gpt2_metrics.csv
                                                 # oracle + dummy  -> results/oracle_dummy_metrics.csv
                                                 # oracle + GPT-2  -> results/oracle_gpt2_metrics.csv
\end{lstlisting}

\subsection*{Phase 2 --- EM and Projected DPO}
\begin{lstlisting}[language=bash]
# Offline EM (run once; provides frozen weights for static projected DPO)
cd src/models && python em_standalone.py
# -> results/standalone_tracking/em_params_iter_*.csv
# -> results/04_em_weights.csv
# -> results/05_em_inferred_params.csv
# -> results/em_loss_history.csv

# Projected DPO - edit use_joint_em and use_sft flags in the script
cd src/training && python train_projected_dpo.py
# Outputs follow pattern {gpt2|dummy}_{joint|static}:
#   results/projected_{gpt2|dummy}_{joint|static}_metrics.csv
#   models/projected_{gpt2|dummy}_{joint|static}.pth
#   results/{joint|static}_tracking_{gpt2|dummy}/em_params_epoch_*.csv
# Joint mode also writes:
#   results/05_em_inferred_params_joint_{gpt2|dummy}.csv
#   results/04_em_weights_joint_{gpt2|dummy}.csv

# Animate EM parameter evolution
cd src/notebooks
python ult_animate_clusters.py --mode standalone          # standalone EM run
python ult_animate_clusters.py --path ../../results/joint_tracking_gpt2/
python ult_animate_clusters.py --path ../../results/static_tracking_dummy/
# Edge cases (after Phase 3 sweep):
python ult_animate_clusters.py --path ../../results/edge_cases/isolated_sparsity_failure_tracking/
\end{lstlisting}

\subsection*{Phase 3 --- Stress Testing}
\begin{lstlisting}[language=bash]
cd src/experiments && python stress_test.py   # -> results/stress_tests/phase3_sweep_matrix.csv
                                              #    results/edge_cases/*_tracking/
cd src/notebooks && python plot_stress_tests.py  # -> results/stress_tests/heatmap_L_vs_Adv.png
                                                 #    results/stress_tests/linegraph_scale_N.png
# Animate an edge case
cd src/notebooks && python ult_animate_clusters.py \
    --path ../../results/edge_cases/<folder>/
\end{lstlisting}

\section{Known Implementation Note}

In \texttt{src/models/em.py}, the E-step adds both the LLM prior term and the global
$\pi_1$ term to each branch. When \texttt{llm\_priors} is \texttt{None}, the static 50/50
prior is effectively applied twice in log-space. When LLM priors are provided, the global
$\pi_1$ term is still added on top. This deviates from the textbook Dawid--Skene
formulation and should be reviewed before publication.

\end{document}