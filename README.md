# EchoNext ProtoECGNet Adaptation

 The goal is to create a baseline for a mini descriptive study, focusing on prototype interpretability for 11 SHD binary flags. The implementation follows the original codebase closely, with customizations for EchoNext's data format (.npy waveforms and metadata CSV). It uses a single 2D global prototype branch for diffuse SHD patterns, with plans to add morphology for local features.
The original ProtoECGNet codebase is available at: https://github.com/bbj-lab/protoecgnet
 This adaptation draws heavily from files like proto_models2D.py (for the 2D branch), ecg_utils.py (for data handling inspiration), label_co.py (for cooccurrence matrix), and main.py (for training logic)
 
## Setup
Environment: Use Python 3.10+ with dependencies: torch, numpy, pandas, scikit-learn, scipy. Install via pip install torch numpy pandas scikit-learn scipy.
Data Preparation: Place EchoNext files in /opt/gpudata/ecg/echonext/ (metadata CSV and split .npy waveforms). 
Run the Code: Use echonext_protoecg_adaptation.py--mode train to train, --mode evaluate for test metrics, --mode project for prototype projection. Outputs save to /opt/gpudata/summereunann/echonext_experiments/.

## Data Handling
The code loads the EchoNext metadata CSV to extract splits (train: ~72k samples, val: ~4.6k, test: ~5.4k) and the 11 binary SHD labels. For waveforms, it uses np.load to read split-specific .npy files (shaped (N,1,2500,12)). Each ECG is squeezed to (2500,12), downsampled to 100 Hz via linear interpolation (resulting in 1000 samples), and transposed to (12,1000) for 2D input. This mirrors the original repo's handling of PTB-XL data in ecg_utils.py, but replaces WFDB record loading with np.load for .npy compatibility. A Jaccard co-occurrence matrix is computed on train labels only (avoiding leakage) and saved as .npy, identical to label_co.py in the repo.
Preprocessing
Preprocessing applies a 0.5 Hz high-pass Butterworth filter per lead to remove baseline wander, matching the paper's Section 3.2 and the repo's ecg_utils.py. Global standardization is fitted on a sample of train data (up to 5,000 ECGs flattened across leads and time) to normalize the entire dataset, ensuring consistent latent features. This is performed in a custom transform class, similar to the repo's standardize function, and applied during dataloading for efficiency.


## Model Architecture
The model uses a single 2D global prototype branch, adapted from proto_models2D.py in the repo and the paper's Section 3.3. The backbone is a modified ResNet18 with conv2d layers treating leads as height and time as width, including proper residual blocks with shortcuts and ReLU activations. Add-on layers (two linear+ReLU) refine the latent space to 512 dimensions. The prototype head has learnable vectors (5 per SHD label, total 55), computing scaled positive cosine similarities (paper Eq. 1). A linear classifier on similarities outputs logits for the 11 labels (~12M total parameters). The forward pass takes (B,1,12,1000) inputs and returns logits (B,11), similarities (B,55), and features for projection, aligning with the repo's global branch design.


## Loss and Training
The loss is the full composite from the paper's Eq. 2: BCEWithLogitsLoss for multi-label SHD + clustering (Eq. 4, attracting to positive prototypes) + separation (Eq. 5, repelling negatives) + diversity (Eq. 6, orthogonality via Frobenius norm) + contrastive (Eq. 7, using expanded co-occurrence for prototype pairs). Terms are weighted as in the code and include clamps to bound values for stability. Training replicates the joint stage from the repo's main.py: Adam optimizer, gradient clipping (1.0), ReduceLROnPlateau scheduler (patience 10), and early stopping (patience 20) on val loss. Logging includes per-epoch total loss, components, and val macro-AUROC (handling edge cases like all-0 labels), similar to inference_fusion.py.
Results
Training on the EchoNext splits was stable, completing 42 epochs before early stopping (val loss plateaued at ~ -38.85). Validation macro-AUROC started at 0.6955 (epoch 1) and peaked at 0.7494 (epoch 40), with final components like CE 0.2579 (decreasing classification error), clustering −3.2652 (strong prototype attraction), separation 5.9011 (moderate repulsion), diversity 0.0006 (near-orthogonal prototypes), and contrastive −0.1307 (co-occurrence alignment). This provides a reasonable baseline for SHD prototype analysis (comparable to early EchoNext benchmarks ~0.7-0.8), though below full SOTA (~0.9), indicating useful but improvable embeddings.
Future Plans

## Run prototype projection and manually review top-K matches for SHD interpretability (e.g., patterns in valve flags).
Export frozen embeddings for linear probe on SHD to assess quality.
Analyze prototype drift over age/year for ~8.3k multi-timepoint patients.
Add pos-weighted BCE for imbalance, checkpoint by AUROC, stronger diversity penalty.
Report per-label AUROC/AUPRC with calibrated thresholds.
Extend to morphology branch (partial prototypes) if local features needed.






### Experimental Planning and Evaluations

# Experiment 1: Evaluate Pre-Trained PTB-XL Model on EchoNext (Correlation Check)

### Hypothesis
PTB-XL embeddings and prototypes, trained on direct ECG classification labels, will show moderate correlations with EchoNext SHD labels due to overlapping patterns (e.g., hypertrophy in PTB-XL may align with low LVEF in SHD). This tests indirect transferability without retraining, hypothesizing correlations >0.3 for key labels like valve diseases, revealing if PTB-XL captures SHD signals implicitly.

### Methods and Setup
- Load pre-trained ProtoECGNet weights from Sahil's shared PTB-XL model (using the multi-branch or 2D global branch for diffuse patterns).
- Extract embeddings and prototype activations from EchoNext dataset (all splits: train ~72k, val ~4.6k, test ~5.4k) via forward pass (no gradients, batch size 64 on GPU).
- Use the adapted data loader from echonext_protoecg_adaptation.py to process EchoNext .npy waveforms and metadata, applying the same preprocessing (downsampling to 100 Hz, high-pass filter, standardization).
- Compute correlations between activations/embeddings and SHD flags; train a linear probe (sklearn LogisticRegression) on train embeddings for SHD prediction, evaluate on val/test.
- Code in a Jupyter notebook (experiments/experiment_1.ipynb) for reproducibility, with results saved to /results/experiment_1/.

### Controls and Baselines
- Null baseline: Random embeddings (same dimension as PTB-XL, e.g., Gaussian noise) to check if correlations/AUROC exceed chance (~0.5 AUROC, ~0 correlation).
- Alternative baseline: Simple ResNet18 (without prototypes) pre-trained on PTB-XL, to isolate the value of prototypes vs. standard features.
- If correlations low, subset to PTB-XL labels overlapping SHD (e.g., rhythm/morphology groups from Appendix H).

### Evaluations and Metrics with References
- Primary: Spearman correlations per SHD label (with p-values; reference: scipy.stats.spearmanr), aiming for >0.3 on average.
- Secondary: Linear probe macro-AUROC (sklearn roc_auc_score, macro average, bootstrapped 95% CIs with 1,000 samples), compared to EchoNext Nature paper SOTA ~0.8 and random ~0.5.
- Visual: t-SNE plots of embeddings colored by SHD flags/severity (using sklearn.manifold.TSNE, matplotlib), with clusters inspected for separation.
- Full curves: N/A for zero-shot, but probe training loss/AUROC over iterations if multi-epoch.
- Insights: Focus on interpretability, e.g., "Correlation 0.45 for LVEF flag due to hypertrophy prototypes, improving 20% over random baseline."

### Iteration Strategy
- If correlations <0.3 overall, iterate by subsetting to specific PTB-XL branches (e.g., morphology only) or normalizing embeddings differently.
- If AUROC <0.7, tune probe hyperparameters (e.g., regularization) as noted; revise plan via GitHub issue if major changes needed (e.g., partial fine-tuning).
- Risks like low overlap mitigated with qualitative viz; document revisions in updated md file.

### Timeline and Outputs
- Timeline: 1-2 days (today: setup/extraction; tomorrow: analysis/review).
- Outputs: Notebook with correlations, AUROC (with CIs), t-SNE plots, full logs; PR for code; issue for feedback before next experiment.



This avoids ad-hoc work and provides a full end-to-end view. For example, to determine if results are "good" or "bad," plans specify baselines (e.g., random embeddings AUROC ~0.5) and comparisons (e.g., to the EchoNext Nature paper's SOTA ~0.8). Metrics like loss values are presented with context: full training curves (via TensorBoard or matplotlib plots in notebooks), confidence intervals (bootstrapped for AUROC), per-label breakdowns, and trends over epochs. Static snapshots are avoided; instead, focus on interpretable insights (e.g., "AUROC improved 5% over baseline due to contrastive term"). Plans are self-contained and reviewable via GitHub issues before execution, allowing for feedback and reducing imprecision. Iteration is expected but minimized through upfront documentation—e.g., if AUROC <0.7, tune hyperparameters as noted in the plan.

### Alignment with Initial Experiment Focus
Following discussions, the first experiment prioritizes evaluating the pre-trained PTB-XL model on EchoNext to assess correlations between its embeddings/prototypes and SHD labels, without any new training. This zero-shot baseline checks for indirect links (e.g., PTB-XL hypertrophy patterns correlating with low LVEF in SHD). No further modeling or training occurs until this is complete, reviewed, and informs next steps. This corrects any prior missteps where training was initiated prematurely.

### Code Reproducibility and Shareability
Code is written for full reproducibility: the repo includes the complete codebase, with all changes via incremental pull requests (PRs) describing what was added, why, and how it ties to plans. Pre-commit hooks are configured (via `.pre-commit-config.yaml` with black for formatting, flake8 for linting, and isort for imports) and installed with `pre-commit install` to enforce quality on commits. PRs include code diffs, tests (e.g., unit tests for data loading), and notebooks. This eliminates non-shareable server-based work—collaborators review directly on GitHub without needing in-depth chat discussions or server access. Code reviews are conducted on PRs to catch issues early.

### Shifting from Code Discussions to Repo-Based Reviews
Code details are not discussed extensively in updates or chats; instead, the repo is the source of truth. PRs and issues handle reviews, with the README providing an overview of how to run and navigate the code. This keeps communications high-level and focused on plans/results.

### Presenting Metrics
Metrics are always contextualized: include references (e.g., to paper benchmarks), curves/plots (e.g., loss/AUROC over epochs), and baseline comparisons (e.g., ablation without contrastive). For instance, AUROC reports include CIs, per-label details, and trends to convey model performance meaningfully.

### Data Splitting
The code uses EchoNext's provided splits from the metadata CSV ('split' column): train (72,475 samples), val (4,626), test (5,442). These are the dataset's predefined divisions, ensuring consistency and no leakage—not custom splits.

## Setup
### Environment
Use Python 3.10+ with dependencies: torch, numpy, pandas, scikit-learn, scipy. Install via `pip install torch numpy pandas scikit-learn scipy`. These match the core requirements from the original repo's environment.yml, focusing on PyTorch for modeling and SciPy for signal processing.

### Data Preparation
Place the EchoNext files in `/opt/gpudata/ecg/echonext/` (metadata CSV and split-specific .npy waveforms, e.g., `EchoNext_train_waveforms.npy`). The metadata CSV should include the 'split' column for train/val/test assignment and the 11 SHD flag columns as binary labels. Ensure file permissions allow access; no additional preprocessing is needed beyond what's handled in the code.

### Run the Code
Use `echonext_protoecg_adaptation.py` as the main script. Execute with `--mode train` to train the model (saves checkpoints to `/opt/gpudata/summereunann/echonext_experiments/`), `--mode evaluate` to compute test metrics (e.g., macro-AUROC), or `--mode project` for prototype projection (saves .npz file of mapped ECG segments). Example: `python echonext_protoecg_adaptation.py --mode train`. Outputs include model checkpoints, loss logs, and projected prototypes for analysis.

## Data Handling
The code loads the EchoNext metadata CSV to extract splits (train: ~72k samples, val: ~4.6k, test: ~5.4k) and the 11 binary SHD labels. For waveforms, it uses np.load to read split-specific .npy files (shaped (N,1,2500,12)). Each ECG is squeezed to (2500,12), downsampled to 100 Hz via linear interpolation (resulting in 1000 samples), and transposed to (12,1000) for 2D input. This mirrors the original repo's handling of PTB-XL data in ecg_utils.py, but replaces WFDB record loading with np.load for .npy compatibility. A Jaccard co-occurrence matrix is computed on train labels only (avoiding leakage) and saved as .npy, identical to label_co.py in the repo.

## Preprocessing
Preprocessing applies a 0.5 Hz high-pass Butterworth filter per lead to remove baseline wander, matching the paper's Section 3.2 and the repo's ecg_utils.py. Global standardization is fitted on a sample of train data (up to 5,000 ECGs flattened across leads and time) to normalize the entire dataset, ensuring consistent latent features. This is performed in a custom transform class, similar to the repo's standardize function, and applied during dataloading for efficiency.

## Model Architecture
The model uses a single 2D global prototype branch, adapted from proto_models2D.py in the repo and the paper's Section 3.3. The backbone is a modified ResNet18 with conv2d layers treating leads as height and time as width, including proper residual blocks with shortcuts and ReLU activations. Add-on layers (two linear+ReLU) refine the latent space to 512 dimensions. The prototype head has learnable vectors (5 per SHD label, total 55), computing scaled positive cosine similarities (paper Eq. 1). A linear classifier on similarities outputs logits for the 11 labels (~12M total parameters). The forward pass takes (B,1,12,1000) inputs and returns logits (B,11), similarities (B,55), and features for projection, aligning with the repo's global branch design.

## Loss and Training
The loss is the full composite from the paper's Eq. 2: BCEWithLogitsLoss for multi-label SHD + clustering (Eq. 4, attracting to positive prototypes) + separation (Eq. 5, repelling negatives) + diversity (Eq. 6, orthogonality via Frobenius norm) + contrastive (Eq. 7, using expanded co-occurrence for prototype pairs). Terms are weighted as in the code and include clamps to bound values for stability. Training replicates the joint stage from the repo's main.py: Adam optimizer, gradient clipping (1.0), ReduceLROnPlateau scheduler (patience 10), and early stopping (patience 20) on val loss. Logging includes per-epoch total loss, components, and val macro-AUROC (handling edge cases like all-0 labels), similar to inference_fusion.py.

## Results
Training on the EchoNext splits was stable, completing 42 epochs before early stopping (val loss plateaued at ~ -38.85). Validation macro-AUROC started at 0.6955 (epoch 1) and peaked at 0.7494 (epoch 40), with final components like CE 0.2579 (decreasing classification error), clustering −3.2652 (strong prototype attraction), separation 5.9011 (moderate repulsion), diversity 0.0006 (near-orthogonal prototypes), and contrastive −0.1307 (co-occurrence alignment). This provides a reasonable baseline for SHD prototype analysis (comparable to early EchoNext benchmarks ~0.7-0.8), though below full SOTA (~0.9), indicating useful but improvable embeddings.

## Future Plans
- Run prototype projection and manually review top-K matches for SHD interpretability (e.g., patterns in valve flags).
- Export frozen embeddings for linear probe on SHD to assess quality.
- Analyze prototype drift over age/year for ~8.3k multi-timepoint patients.
- Add pos-weighted BCE for imbalance, checkpoint by AUROC, stronger diversity penalty.
- Report per-label AUROC/AUPRC with calibrated thresholds.
- Extend to morphology branch (partial prototypes) if local features needed.

