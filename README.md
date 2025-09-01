# EchoNext ProtoECGNet Adaptation

## Project Description
The goal is to create a baseline for a mini descriptive study, focusing on prototype interpretability for 11 SHD binary flags. The implementation follows the original codebase closely, with customizations for EchoNext's data format (.npy waveforms and metadata CSV). It uses a single 2D global prototype branch for diffuse SHD patterns, with plans to add morphology for local features.

The original ProtoECGNet codebase is available at: https://github.com/bbj lab/protoecgnet  
This adaptation draws heavily from files like proto_models2D.py (for the 2D branch), ecg_utils.py (for data handling inspiration), label_co.py (for cooccurrence matrix), and main.py (for training logic).

## Setup
### Environment
Use Python 3.10+ with dependencies: torch, numpy, pandas, scikit-learn, scipy. Install via `pip install torch numpy pandas scikit learn scipy`.

### Data Preparation
Place EchoNext files in /opt/gpudata/ecg/echonext/ (metadata CSV and split .npy waveforms).

### Run the Code
Use echonext_protoecg_adaptation.py   mode train to train,   mode evaluate for test metrics,   mode project for prototype projection. Outputs save to /opt/gpudata/summereunann/echonext_experiments/.

## Repository Structure
  **README.md**: This file—project overview, setup, and plans.
  **data_analytics.ipynb**: Notebook for data investigation (e.g., label distributions, sample ECG visualizations).
  **echonext_protoecg_adaptation.py**: Main script for data loading, model, training, evaluation, and projection.
  **requirements.txt**: Dependency list for reproduction.

## Experimental Planning and Evaluations
All experiments are planned in advance with structured documentation to avoid ad hoc work and ensure clear evaluations. Plans are written as Markdown files in the repo's experiments/ directory (e.g., experiment_1.md) and discussed via GitHub issues for feedback before execution. Each plan includes a hypothesis, methods/setup, controls/baselines, evaluations/metrics with references, iteration strategy for revisions, and timeline/outputs. This reduces confusion, imprecision, and repeated revisions by providing a full end to end view upfront.

For example, loss values are always presented with context, including full training curves (via TensorBoard or matplotlib plots), comparisons to baselines (e.g., random model AUROC 0.5), and references (e.g., to the EchoNext Nature paper's ~0.8 SOTA). Metrics like AUROC include confidence intervals (bootstrapped) and per label breakdowns.

### Experiment 1: Evaluate Pre Trained PTB XL Model on EchoNext
#### Hypothesis
Prototypes learned on standard ECG tasks (e.g., PTB XL labels) will show moderate correlations with SHD outcomes in EchoNext, as indirect features like hypertrophy or conduction delays may align with structural abnormalities.  
Scaling the prototype method to larger datasets with lower quality labels (e.g., machine generated or fewer classes) will improve embedding generalizability, following data scaling laws where increased data size compensates for label noise, leading to better performance on SHD prediction (measured by embedding quality) while maintaining strong results on PTB XL for direct classification.  
Moving toward self supervised prototype approaches will further enhance embeddings for unseen tasks like SHD, by leveraging unlabeled data to learn more robust representations, potentially matching or exceeding supervised SOTA on embedding quality metrics.

#### Experimental Design
The design follows a phased, iterative approach with clear controls, evaluations, and documentation to address feedback on planning. Experiments will be documented in the repo's experiments/ folder as Markdown files (e.g., experiment_1.md), reviewed via GitHub issues/PRs before execution. We'll use the 4 datasets mentioned (assuming PTB XL, EchoNext, and 2 others like MIMIC IV ECG/Harvard Emory with varying size/quality) for scaling tests. No new modeling until plans are approved. Timeline: Phase 1 (1 week), Phase 2 (2 weeks), Phase 3 (ongoing iteration).

**Phase 1: Baseline Correlation Check (No Training)**  
  Methods/Setup: Load pre trained ProtoECGNet weights from PTB XL (Sahil's shared model). Extract embeddings and prototype activations from EchoNext (all splits: train ~72k, val ~4.6k, test ~5.4k) via forward pass (no gradients, batch 64 on GPU). Use adapted loader for preprocessing (downsample to 100 Hz, filter, standardize). Compute Spearman correlations between activations/embeddings and SHD flags; train scikit learn LogisticRegression probe on train embeddings for SHD prediction, evaluate on val/test.  
  Controls/Baselines: Random embeddings (Gaussian noise, same dim) as null (expected AUROC ~0.5, correlations ~0); non prototype ResNet (PTB XL pre trained) for comparison.  
  Evaluations/Metrics: Spearman rho/p values per label (scipy.stats, aim >0.3 moderate); macro/per label AUROC (sklearn roc_auc_score, with bootstrapped CIs vs. EchoNext paper SOTA ~0.8/random 0.5); t SNE viz of embeddings by SHD (sklearn.manifold, inspect clusters). Full curves N/A for zero shot; report in notebook with plots/heatmaps.  
  Iteration/Risks: If correlations <0.3, subset to PTB XL branches (e.g., morphology only); mitigate low signal with qualitative viz. Revise via issue if major (e.g., add dimensionality reduction before probe).  
  Outputs/Timeline: Notebook with results/plots, saved to /results/experiment_1/; 1 2 days (extraction today, analysis tomorrow).

**Phase 2: Prototype Application and Descriptive Study on EchoNext**  
  Methods/Setup: Apply ProtoECGNet to EchoNext (start with PTB XL pre trained, fine tune if correlations from Phase 1 warrant). Use 2D global branch for diffuse SHD; project prototypes to real ECG segments (repo's push.py style).  
  Controls/Baselines: Scratch trained model on EchoNext; ablations (no contrastive) to isolate prototype value.  
  Evaluations/Metrics: Prototype quality via manual review (paper's rubric: representativeness/clarity on top K matches); embedding quality via linear probe AUROC (as Phase 1); loss curves/components over epochs (TensorBoard exports, reference paper's PTB XL loss trends). Compare to Phase 1 zero shot.  
  Iteration/Risks: If prototypes not meaningful, increase per class count or add morphology; mitigate imbalance with weighted BCE.  
  Outputs/Timeline: Descriptive report with projected ECG viz; 3 5 days after Phase 1.

**Phase 3: Generalization and Scaling Experiments**  
  Methods/Setup: Explore scaling with 4 datasets (PTB XL high quality/small, EchoNext medium, larger/low quality like MIMIC IV ECG/machine generated). Fine tune prototypes; test self supervised variants (e.g., contrastive pre training on unlabeled ECGs).  
  Controls/Baselines: Small scale only (low data/high quality) vs. large scale (high data/low quality); supervised vs. self supervised.  
  Evaluations/Metrics: Scaling curves (AUROC vs. data size/label quality, reference data scaling laws papers like Kaplan et al. 2020); PTB XL AUROC for direct eval (~0.91 reference), EchoNext embedding AUROC for quality (~0.8 reference); per label metrics with CIs.  
  Iteration/Risks: If scaling doesn't hold, adjust label noise levels; mitigate compute with subsampling.  
  Outputs/Timeline: Plots/notebooks on scaling; ongoing after Phase 2, 1 2 weeks per variant.

This design ensures controls (random/non proto), clear evals (curves, references like paper AUROC), and documentation to reduce confusion. Feedback welcome via issue!

## hypothesis
Our overarching hypothesis is that prototype based embeddings from ECG models like ProtoECGNet can capture latent patterns useful for tasks beyond direct ECG classification, such as predicting structural heart disease (SHD) labels from the EchoNext dataset, where clinical correlations are not well understood. Specifically, we hypothesize that:
  Prototypes learned on standard ECG tasks (e.g., PTB XL labels) will show moderate correlations with SHD outcomes in EchoNext, as indirect features like hypertrophy or conduction delays may align with structural abnormalities.
  Scaling the prototype method to larger datasets with lower quality labels (e.g., machine generated or fewer classes) will improve embedding generalizability, following data scaling laws where increased data size compensates for label noise, leading to better performance on SHD prediction (measured by embedding quality) while maintaining strong results on PTB XL for direct classification for ECG abnormalities.
  Moving toward self supervised prototype approaches will further enhance embeddings for unseen tasks like SHD, by leveraging unlabeled data to learn more robust representations, potentially matching or exceeding supervised SOTA on embedding quality metrics.

## Experimental Design
The experimental design is structured in phases with clear hypotheses, methods, controls, evaluations (including metrics with references and baselines), iteration strategies, and timelines to ensure a full end to end view and reduce confusion. Plans are documented in the repo's experiments/ folder and reviewed via GitHub issues/PRs before execution. We'll use the 4 datasets mentioned (PTB XL, EchoNext, MIMIC IV ECG, Harvard Emory) for scaling, with size inversely proportional to label quality (e.g., PTB XL small/high quality, MIMIC large/low quality). No new modeling until approved.

### Phase 1: Baseline Correlation Check (No Training   Zero Shot PTB XL on EchoNext)
  **Hypothesis**: PTB XL embeddings/prototypes will show moderate correlations (>0.3) with SHD labels, as indirect ECG patterns (e.g., hypertrophy) align with structural changes.
  **Methods/Setup**: Load PTB XL weights (Sahil's shared .pth); extract embeddings/activations from EchoNext (all splits) via forward pass (no grad, batch 64 on GPU). Use adapted loader for preprocessing. Compute correlations; train linear probe on train embeddings for SHD prediction.
  **Controls/Baselines**: Random embeddings (Gaussian, same dim) as null (~0.5 AUROC, ~0 correlation); non prototype ResNet (PTB XL pre trained) for comparison.
  **Evaluations/Metrics**: Spearman rho/p values per label (scipy.stats, aim >0.3); macro/per label AUROC (sklearn roc_auc_score, bootstrapped CIs vs. EchoNext paper ~0.8 SOTA/random 0.5); t SNE viz (sklearn.manifold) with clusters by SHD, heatmaps for correlations. Report in notebook with plots.
  **Iteration/Risks**: If <0.3, subset PTB XL branches (e.g., morphology); mitigate low overlap with viz. Revise via issue.
  **Timeline/Outputs**: 1 2 days; notebook with results/plots in /results/experiment_1/, PR for code.

### Phase 2: Prototype Application and Descriptive Study on EchoNext
  **Hypothesis**: Fine tuning PTB XL prototypes on EchoNext will surface meaningful SHD patterns, with embeddings outperforming Phase 1 zero shot on probe AUROC.
  **Methods/Setup**: Fine tune 2D global branch on EchoNext SHD (start from PTB XL weights if Phase 1 shows promise); project prototypes to ECG segments.
  **Controls/Baselines**: Scratch trained on EchoNext; ablations (no contrastive) for prototype value.
  **Evaluations/Metrics**: Manual prototype review (paper rubric for representativeness/clarity on top K matches); probe AUROC (as Phase 1); loss curves (TensorBoard, ref paper PTB XL trends ~0.91 AUROC). Compare to Phase 1.
  **Iteration/Risks**: If not meaningful, add morphology; weighted BCE for imbalance. Issue for revisions.
  **Timeline/Outputs**: 3 5 days after Phase 1; report with viz in notebook.

### Phase 3: Generalization and Scaling with Self Supervised Approaches
  **Hypothesis**: Scaling to larger/low quality datasets improves embeddings (AUROC gain ~10 20% per data doubling, per scaling laws), and self supervised variants match supervised SOTA (~0.9) on embedding quality.
  **Methods/Setup**: Fine tune on 4 datasets; test self supervised (e.g., contrastive pre train on unlabeled ECGs).
  **Controls/Baselines**: Small/high quality vs. large/low quality; supervised vs. self supervised.
  **Evaluations/Metrics**: Scaling curves (AUROC vs. size/quality, ref Kaplan et al. 2020); PTB XL direct AUROC (~0.91 ref), EchoNext probe AUROC (~0.8 ref); per label with CIs.
  **Iteration/Risks**: If no gain, adjust noise; subsampling for compute. Issue for revisions.
  **Timeline/Outputs**: Ongoing after Phase 2; plots/notebooks per variant, 1 2 weeks each.

Feedback welcome via issue!
