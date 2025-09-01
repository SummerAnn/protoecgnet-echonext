# Experiment 1: Evaluate Pre-Trained PTB-XL Model on EchoNext (Correlation Check)

## Hypothesis
Prototypes learned on standard ECG tasks (e.g., PTB-XL labels) will show moderate correlations (>0.3) with SHD outcomes in EchoNext, as indirect features like hypertrophy or conduction delays may align with structural abnormalities. Scaling the prototype method to larger datasets with lower-quality labels (e.g., machine-generated or fewer classes) will improve embedding generalizability, following data scaling laws where increased data size compensates for label noise, leading to better performance on SHD prediction (measured by embedding quality) while maintaining strong results on PTB-XL for direct classification. Moving toward self-supervised prototype approaches will further enhance embeddings for unseen tasks like SHD, by leveraging unlabeled data to learn more robust representations, potentially matching or exceeding supervised SOTA on embedding quality metrics.

## Methods and Setup
- Load pre-trained ProtoECGNet weights from Sahil's shared PTB-XL model (using the multi-branch or 2D global branch for diffuse patterns; model class definition adapted from [src/proto_models2D.py](https://github.com/bbj-lab/protoecgnet/blob/main/src/proto_models2D.py) in the original repo for the 2D branch and [src/backbones.py](https://github.com/bbj-lab/protoecgnet/blob/main/src/backbones.py) for the ResNet2D backbone).
- Extract embeddings and prototype activations from EchoNext dataset (all splits: train ~72k, val ~4.6k, test ~5.4k) via forward pass (no gradients, batch size 64 on GPU), using the model's forward method (from [src/proto_models2D.py](https://github.com/bbj-lab/protoecgnet/blob/main/src/proto_models2D.py)).
- Use the adapted data loader from [echonext_protoecg_adaptation.py](echonext_protoecg_adaptation.py) to process EchoNext .npy waveforms and metadata, applying the same preprocessing (downsampling to 100 Hz, high-pass filter, standardization; filter inspired by [src/ecg_utils.py](https://github.com/bbj-lab/protoecgnet/blob/main/src/ecg_utils.py)).
- Compute Spearman correlations between activations/embeddings and SHD flags; train a linear probe (sklearn LogisticRegression) on train embeddings for SHD prediction, evaluate on val/test.
- Code in [data_analytics.ipynb](data_analytics.ipynb) for reproducibility, with results saved to /results/experiment_1/.

## Controls and Baselines
- Null baseline: Random embeddings (Gaussian noise, same dimension as PTB-XL latent space from [src/proto_models2D.py](https://github.com/bbj-lab/protoecgnet/blob/main/src/proto_models2D.py), typically 512-dim) to check if correlations/AUROC exceed chance (~0.5 AUROC, ~0 correlation).
- Alternative baseline: Simple ResNet18 (without prototypes) pre-trained on PTB-XL, using the backbone from [src/backbones.py](https://github.com/bbj-lab/protoecgnet/blob/main/src/backbones.py), to isolate the value of prototypes vs. standard features.
- If correlations low, subset to PTB-XL labels overlapping SHD (e.g., rhythm/morphology groups from [scp_statementsRegrouped2.csv](https://github.com/bbj-lab/protoecgnet/blob/main/scp_statementsRegrouped2.csv)).

## Evaluations and Metrics with References
- Primary: Spearman correlations per label (with p-values; using scipy.stats.spearmanr), aiming for >0.3 on average.
- Secondary: Linear probe macro-AUROC (sklearn roc_auc_score, macro average, bootstrapped 95% CIs with 1,000 samples), compared to EchoNext Nature paper SOTA ~0.8 [](https://www.nature.com/articles/s41586-025-09227-0) and random ~0.5.
- Visual: t-SNE plots of embeddings colored by SHD flags/severity (using sklearn.manifold.TSNE, matplotlib), with clusters inspected for separation.
- Full curves: N/A for zero-shot, but probe training loss/AUROC over iterations if multi-epoch.
- Insights: Focus on interpretability, e.g., "Correlation 0.45 for LVEF flag due to hypertrophy prototypes, improving 20% over random baseline."

## Iteration Strategy
- If correlations <0.3 overall, iterate by subsetting to specific PTB-XL branches (e.g., morphology only, using code from [src/proto_models2D.py](https://github.com/bbj-lab/protoecgnet/blob/main/src/proto_models2D.py)) or normalizing embeddings differently.
- If AUROC <0.7, tune probe hyperparameters (e.g., regularization) as noted; revise plan via GitHub issue if major changes needed (e.g., partial fine-tuning).
- Risks like low overlap mitigated with qualitative viz; document revisions in updated md file.

## Timeline and Outputs
- Timeline: 1-2 days (today: setup/extraction; tomorrow: analysis/review).
- Outputs: Notebook with correlations, AUROC (with CIs), t-SNE plots, full logs; PR for code; issue for feedback before next experiment.
