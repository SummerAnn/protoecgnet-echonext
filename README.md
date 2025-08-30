EchoNext ProtoECGNet Adaptation

 The goal is to create a baseline for a mini descriptive study, focusing on prototype interpretability for 11 SHD binary flags. The implementation follows the original codebase closely, with customizations for EchoNext's data format (.npy waveforms and metadata CSV). It uses a single 2D global prototype branch for diffuse SHD patterns, with plans to add morphology for local features.
The original ProtoECGNet codebase is available at: https://github.com/bbj-lab/protoecgnet
 This adaptation draws heavily from files like proto_models2D.py (for the 2D branch), ecg_utils.py (for data handling inspiration), label_co.py (for cooccurrence matrix), and main.py (for training logic)
 
Setup
Environment: Use Python 3.10+ with dependencies: torch, numpy, pandas, scikit-learn, scipy. Install via pip install torch numpy pandas scikit-learn scipy.
Data Preparation: Place EchoNext files in /opt/gpudata/ecg/echonext/ (metadata CSV and split .npy waveforms). 
Run the Code: Use echonext_protoecg_adaptation.py--mode train to train, --mode evaluate for test metrics, --mode project for prototype projection. Outputs save to /opt/gpudata/summereunann/echonext_experiments/.

Data Handling
The code loads the EchoNext metadata CSV to extract splits (train: ~72k samples, val: ~4.6k, test: ~5.4k) and the 11 binary SHD labels. For waveforms, it uses np.load to read split-specific .npy files (shaped (N,1,2500,12)). Each ECG is squeezed to (2500,12), downsampled to 100 Hz via linear interpolation (resulting in 1000 samples), and transposed to (12,1000) for 2D input. This mirrors the original repo's handling of PTB-XL data in ecg_utils.py, but replaces WFDB record loading with np.load for .npy compatibility. A Jaccard co-occurrence matrix is computed on train labels only (avoiding leakage) and saved as .npy, identical to label_co.py in the repo.
Preprocessing
Preprocessing applies a 0.5 Hz high-pass Butterworth filter per lead to remove baseline wander, matching the paper's Section 3.2 and the repo's ecg_utils.py. Global standardization is fitted on a sample of train data (up to 5,000 ECGs flattened across leads and time) to normalize the entire dataset, ensuring consistent latent features. This is performed in a custom transform class, similar to the repo's standardize function, and applied during dataloading for efficiency.


Model Architecture
The model uses a single 2D global prototype branch, adapted from proto_models2D.py in the repo and the paper's Section 3.3. The backbone is a modified ResNet18 with conv2d layers treating leads as height and time as width, including proper residual blocks with shortcuts and ReLU activations. Add-on layers (two linear+ReLU) refine the latent space to 512 dimensions. The prototype head has learnable vectors (5 per SHD label, total 55), computing scaled positive cosine similarities (paper Eq. 1). A linear classifier on similarities outputs logits for the 11 labels (~12M total parameters). The forward pass takes (B,1,12,1000) inputs and returns logits (B,11), similarities (B,55), and features for projection, aligning with the repo's global branch design.


Loss and Training
The loss is the full composite from the paper's Eq. 2: BCEWithLogitsLoss for multi-label SHD + clustering (Eq. 4, attracting to positive prototypes) + separation (Eq. 5, repelling negatives) + diversity (Eq. 6, orthogonality via Frobenius norm) + contrastive (Eq. 7, using expanded co-occurrence for prototype pairs). Terms are weighted as in the code and include clamps to bound values for stability. Training replicates the joint stage from the repo's main.py: Adam optimizer, gradient clipping (1.0), ReduceLROnPlateau scheduler (patience 10), and early stopping (patience 20) on val loss. Logging includes per-epoch total loss, components, and val macro-AUROC (handling edge cases like all-0 labels), similar to inference_fusion.py.
Results
Training on the EchoNext splits was stable, completing 42 epochs before early stopping (val loss plateaued at ~ -38.85). Validation macro-AUROC started at 0.6955 (epoch 1) and peaked at 0.7494 (epoch 40), with final components like CE 0.2579 (decreasing classification error), clustering −3.2652 (strong prototype attraction), separation 5.9011 (moderate repulsion), diversity 0.0006 (near-orthogonal prototypes), and contrastive −0.1307 (co-occurrence alignment). This provides a reasonable baseline for SHD prototype analysis (comparable to early EchoNext benchmarks ~0.7-0.8), though below full SOTA (~0.9), indicating useful but improvable embeddings.
Future Plans

Run prototype projection and manually review top-K matches for SHD interpretability (e.g., patterns in valve flags).
Export frozen embeddings for linear probe on SHD to assess quality.
Analyze prototype drift over age/year for ~8.3k multi-timepoint patients.
Add pos-weighted BCE for imbalance, checkpoint by AUROC, stronger diversity penalty.
Report per-label AUROC/AUPRC with calibrated thresholds.
Extend to morphology branch (partial prototypes) if local features needed.





