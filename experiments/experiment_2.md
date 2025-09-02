# Experiment 2: Prototype Application and Descriptive Study on EchoNext

## Hypothesis
Fine-tuning PTB-XL prototypes on EchoNext will surface meaningful SHD patterns, with embeddings outperforming Phase 1 zero-shot on probe AUROC.

## Methods/Setup
- Fine-tune 2D global branch on EchoNext SHD (start from PTB-XL weights if Phase 1 shows promise).
- Project prototypes to real ECG segments (repo's push.py style).

## Controls/Baselines
- Scratch-trained on EchoNext; ablations (no contrastive) for prototype value.

## Evaluations/Metrics
- Manual prototype review (paper rubric for representativeness/clarity on top-K matches).
- Probe AUROC (as Phase 1); loss curves (TensorBoard, ref paper PTB-XL trends ~0.91).
- Compare to Phase 1 zero-shot.

## Iteration/Risks
- If not meaningful, add morphology; weighted BCE for imbalance. Issue for revisions.

## Timeline/Outputs
- 3-5 days after Phase 1; report with viz in notebook.
