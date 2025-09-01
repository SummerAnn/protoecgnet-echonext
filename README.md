
# ProtoECGNet-EchoNext Adaptation

## Project Overview
This repository adapts the ProtoECGNet model (from the MLHC 2025 paper and [bbj-lab/protoecgnet](https://github.com/bbj-lab/protoecgnet)) to the EchoNext dataset for a mini descriptive study on structural heart disease (SHD) classification. The focus is on learning meaningful prototype embeddings for tasks beyond direct ECG classification, evaluating on 11 SHD binary flags where clinical correlations are less understood. We start with zero-shot evaluation of pre-trained PTB-XL models on EchoNext to check embedding correlations with SHD, then explore improvements like scaling to larger/lower-quality datasets and self-supervised approaches. PTB-XL serves as the benchmark for direct ECG tasks, EchoNext for embedding quality.

## Setup
### Environment
Use Python 3.10+ with dependencies in [requirements.txt](requirements.txt). Install via:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

### Data Preparation
Place EchoNext files in `/opt/gpudata/ecg/echonext/` (metadata CSV and .npy waveforms). Obtain from [PhysioNet](https://physionet.org/content/echonext/1.0.0/). PTB-XL weights from shared drive (https://drive.google.com/drive/folders/1xYcpSKotubnwnYQfn_7xxlygnq1KmbOO) or repo training.

### Run the Code
Use [echonext_protoecg_adaptation.py](echonext_protoecg_adaptation.py):
```
python echonext_protoecg_adaptation.py --mode train  # Train (outputs to echonext_experiments/)
python echonext_protoecg_adaptation.py --mode evaluate  # Test metrics
python echonext_protoecg_adaptation.py --mode project  # Prototype projection
```

### Running the Notebook
For data investigation and Experiment 1: `jupyter notebook data_analytics.ipynb`.

## Repository Structure
- `README.md`: Project overview and setup.
- `data_analytics.ipynb`: Notebook for data investigation and Experiment 1 (PTB-XL eval).
- `echonext_protoecg_adaptation.py`: Main script for data loading, model, training, evaluation, projection.
- `requirements.txt`: Dependency list.
- `experiments/`: Experiment plans (e.g., [experiment_1.md](experiments/experiment_1.md)).
- `results/`: Outputs like plots, embeddings.
- `.pre-commit-config.yaml`: Hooks for code quality.
- `.gitignore`: Ignores temp files.

## Experiments
Experiments follow structured plans in `experiments/` (reviewed via issues/PRs before running) to ensure clear evaluations and controls. Each includes hypothesis, methods, baselines, metrics with references/comparisons (e.g., curves, CIs vs. baselines like random ~0.5 AUROC or EchoNext SOTA ~0.8), iteration strategy, and timeline. This minimizes revisions and confusion. No training until Experiment 1 (PTB-XL zero-shot on EchoNext) is complete.

### Experiment 1: Evaluate Pre-Trained PTB-XL Model on EchoNext
See [experiments/experiment_1.md](experiments/experiment_1.md) for full plan. Hypothesis: PTB-XL embeddings correlate moderately (>0.3) with SHD. Methods: Extract embeddings/activations, compute correlations, linear probe. Baselines: Random embeddings (~0.5 AUROC). Metrics: Spearman rho, AUROC with CIs (ref EchoNext ~0.8). Timeline: 1-2 days.

## Metrics Presentation
Metrics are contextualized with references (e.g., EchoNext Nature paper ~0.8 AUROC, PTB-XL paper ~0.91 AUROC), curves/plots (e.g., loss/AUROC over epochs via TensorBoard), and baseline comparisons (e.g., random ~0.5 AUROC). For instance, AUROC includes bootstrapped CIs, per-label details, and trends.

## Data Handling
Loads metadata for splits (train ~72k, val ~4.6k, test ~5.4k—EchoNext's provided) and SHD labels. Waveforms via np.load, downsampled to 100 Hz, transposed to (12,1000). Co-occurrence matrix on train.

## Preprocessing
0.5 Hz high-pass filter per lead, global standardization on train sample.

## Model Architecture
2D global branch: Modified ResNet18 backbone, add-on layers, prototypes (5 per label, 55 total), cosine similarities, linear classifier (~12M params).

## Loss and Training
Full loss (BCE + clustering + separation + diversity + contrastive) with clamps. Joint stage: Adam, scheduler, early stopping, logging with AUROC.

## Results
Training stable (42 epochs, val AUROC peak 0.7494). Components: CE 0.2579, clustering −3.2652, etc. Baseline vs. SOTA ~0.9; useful for SHD study.

## Future Plans
- Prototype projection/top-K review.
- Frozen embeddings linear probe.
- Drift analysis on multi-timepoint patients.
- Weighted BCE, AUROC checkpointing, stronger diversity.
- Per-label metrics/calibration.
- Morphology branch/self-supervised extensions.

Feedback/PRs welcome!
```
