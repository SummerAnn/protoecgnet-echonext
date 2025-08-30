#!/usr/bin/env python3
"""
Corrected ProtoECGNet Adaptation for EchoNext Dataset
Multi-Label SHD Classification with Full Prototype Loss

Key Fixes:
- Positive cosine similarity (not negative Euclidean distance)
- Global preprocessing with baseline removal
- Fixed ResNet2D architecture with complete layers and proper shortcuts
- Proper contrastive loss calculation with robust clamps
- Loss component logging and validation metrics
- Fixed prototype projection using cosine similarity
- Efficient projection saving as .npz

Author: Based on Summer's adaptation and original ProtoECGNet
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

# Configuration
ECHONEXT_PATH = "/opt/gpudata/ecg/echonext"
OUTPUT_DIR = "/opt/gpudata/summereunann/echonext_experiments"
WORKING_DIR = "/opt/gpudata/summereunann"
PREPROCESSING_PATH = "/opt/gpudata/summereunann/preprocessing"

MODEL_CONFIG = {
    'num_prototypes_per_class': 5,  # Prototypes per SHD label
    'proto_dim': 512,
    'backbone': 'resnet2d_global',  # Use 2D for SHD
    'dropout': 0.4,
    'l2_reg': 1e-4,
}

TRAINING_CONFIG = {
    'batch_size': 64,
    'epochs': 150,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'patience': 20,
    'gradient_clip': 1.0,
}

DATA_CONFIG = {
    'sampling_rate': 100,  # Downsample from 250
    'remove_baseline': True,
    'standardize': True,
    'use_synthetic_fallback': False,
    'num_workers': 8,
}

LOSS_WEIGHTS = {
    'classification': 1.0,
    'clustering': 0.004,
    'separation': 0.0004,
    'diversity': 250,
    'contrastive': 300,
}

# SHD label columns from dataset (11 specific binary flags)
SHD_LABEL_COLUMNS = [
    'lvef_lte_45_flag',
    'lvwt_gte_13_flag',
    'aortic_stenosis_moderate_or_greater_flag',
    'aortic_regurgitation_moderate_or_greater_flag',
    'mitral_regurgitation_moderate_or_greater_flag',
    'tricuspid_regurgitation_moderate_or_greater_flag',
    'pulmonary_regurgitation_moderate_or_greater_flag',
    'rv_systolic_dysfunction_moderate_or_greater_flag',
    'pericardial_effusion_moderate_large_flag',
    'pasp_gte_45_flag',
    'tr_max_gte_32_flag',
]

NUM_CLASSES = len(SHD_LABEL_COLUMNS)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREPROCESSING_PATH, exist_ok=True)
os.chdir(WORKING_DIR)

# Data Class
class EchoNextDataset(Dataset):
    def __init__(self, data_path, labels_df, transform=None, sampling_rate=100):
        self.data_path = data_path
        self.labels_df = labels_df
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.waveforms = None
        
        # Load the correct waveform file based on split
        split = labels_df['split'].iloc[0] if len(labels_df) > 0 else 'train'
        waveform_path = os.path.join(data_path, f"EchoNext_{split}_waveforms.npy")
        if os.path.exists(waveform_path):
            self.waveforms = np.load(waveform_path)
            print(f"Loaded {split} waveforms: {self.waveforms.shape}")
        else:
            print(f"Warning: Waveform file not found: {waveform_path}")

    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        shd_labels = row[SHD_LABEL_COLUMNS].values.astype(float)
        ecg_data = None
        
        if self.waveforms is not None and idx < len(self.waveforms):
            ecg_data = self.waveforms[idx]
        
        if ecg_data is None:
            if DATA_CONFIG['use_synthetic_fallback']:
                ecg_data = generate_synthetic_ecg(2500, 12)  # Placeholder at 250 Hz
            else:
                raise ValueError(f"No waveform for index {idx}")
        
        # Squeeze and downsample to 100 Hz (1000 samples)
        ecg_data = np.squeeze(ecg_data)  # (2500, 12)
        indices = np.linspace(0, ecg_data.shape[0]-1, 1000, dtype=int)
        ecg_data = ecg_data[indices, :]
        
        if self.transform:
            ecg_data = self.transform(ecg_data)
        
        return torch.FloatTensor(ecg_data.T), torch.FloatTensor(shd_labels)  # (12, 1000), labels vector

def generate_synthetic_ecg(length=2500, num_leads=12):
    """Generate synthetic ECG data for testing"""
    return np.random.randn(1, 1, length, num_leads) * 0.1

# Preprocessing
def remove_baseline_wander(ecg_data, sampling_rate=100, cutoff=0.5, order=1):
    """Remove baseline wander from ECG data"""
    b, a = butter(order, cutoff / (sampling_rate / 2), btype='high')
    for lead in range(ecg_data.shape[0]):
        ecg_data[lead] = filtfilt(b, a, ecg_data[lead])
    return ecg_data

class PreprocessTransform:
    def __init__(self, scaler=None, remove_baseline=True):
        self.scaler = scaler
        self.remove_baseline = remove_baseline

    def __call__(self, ecg_data):
        if self.remove_baseline:
            ecg_data = remove_baseline_wander(ecg_data)
        
        # Global standardize (flatten all)
        ecg_data_reshaped = ecg_data.reshape(-1, 1)
        if self.scaler is None:
            self.scaler = StandardScaler().fit(ecg_data_reshaped)
        ecg_data = self.scaler.transform(ecg_data_reshaped).reshape(12, -1)
        return ecg_data

def load_echonext_data():
    metadata_file = os.path.join(ECHONEXT_PATH, "EchoNext_metadata_100k.csv")
    df = pd.read_csv(metadata_file)
    print(f"Loaded metadata with {len(df)} samples")
    
    # Ensure binary labels are 0/1
    for col in SHD_LABEL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(float)
        else:
            print(f"Warning: Column {col} not found in metadata")
            df[col] = 0.0
    
    return df

# Dataloaders
def get_echonext_dataloaders(batch_size=64, sampling_rate=100):
    df = load_echonext_data()
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    print(f"Dataset splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Fit preprocessor on train data with larger sample
    print("Fitting preprocessor on training data...")
    temp_dataset = EchoNextDataset(ECHONEXT_PATH, train_df, transform=None, sampling_rate=sampling_rate)
    temp_data = []
    for i in range(min(5000, len(temp_dataset))):  # Increased sample size
        try:
            ecg_data, _ = temp_dataset[i]
            temp_data.append(ecg_data.numpy())
        except:
            continue
    
    if temp_data:
        temp_data = np.stack(temp_data)
        preprocessor = PreprocessTransform(remove_baseline=DATA_CONFIG['remove_baseline'])
        # Fit scaler on sample (global standardization)
        temp_data_reshaped = temp_data.reshape(-1, 1)
        preprocessor.scaler = StandardScaler().fit(temp_data_reshaped)
        print("Preprocessor fitted successfully")
    else:
        print("Warning: Could not fit preprocessor, using default")
        preprocessor = PreprocessTransform(remove_baseline=DATA_CONFIG['remove_baseline'])
    
    # Compute co-occurrence matrix on train labels only (avoid data leakage)
    print("Computing co-occurrence matrix on training data...")
    labels = train_df[SHD_LABEL_COLUMNS].values
    cooccur = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            a = labels[:, i] > 0
            b = labels[:, j] > 0
            intersection = np.sum(a & b)
            union = np.sum(a | b)
            cooccur[i, j] = intersection / union if union > 0 else 0
    
    cooccur_path = os.path.join(PREPROCESSING_PATH, 'shd_cooccur.npy')
    np.save(cooccur_path, cooccur)
    print(f"Saved co-occurrence matrix to {cooccur_path}")
    
    # Create datasets with fitted preprocessor
    train_dataset = EchoNextDataset(ECHONEXT_PATH, train_df, preprocessor, sampling_rate)
    val_dataset = EchoNextDataset(ECHONEXT_PATH, val_df, preprocessor, sampling_rate)
    test_dataset = EchoNextDataset(ECHONEXT_PATH, test_df, preprocessor, sampling_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=DATA_CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=DATA_CONFIG['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=DATA_CONFIG['num_workers'])
    
    return train_loader, val_loader, test_loader

# Model (2D Global for SHD) - Complete ResNet2D implementation with proper Block Module
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1,1)):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if stride != (1,1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return nn.ReLU()(self.block(x) + self.shortcut(x))

class ResNet2D(nn.Module):
    def __init__(self, proto_dim=512):
        super(ResNet2D, self).__init__()
        # Fixed conv1 kernel for 12-lead ECG
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1,2), padding=(3,3))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))
        
        # Complete residual blocks with proper downsampling
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=(1,2))
        self.layer3 = self._make_layer(128, 256, 2, stride=(1,2))
        self.layer4 = self._make_layer(256, 512, 2, stride=(1,2))
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, proto_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride=(1,1)):
        layers = [Block(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(Block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch,1,12,1000)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ProtoECGNetSHD(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_prototypes_per_class=MODEL_CONFIG['num_prototypes_per_class'], proto_dim=MODEL_CONFIG['proto_dim']):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_prototypes = num_classes * num_prototypes_per_class
        self.proto_dim = proto_dim
        
        self.feature_extractor = ResNet2D(proto_dim)
        self.add_on_layers = nn.Sequential(
            nn.Linear(proto_dim, proto_dim),
            nn.ReLU(),
            nn.Linear(proto_dim, proto_dim)
        )
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, proto_dim))
        self.classifier = nn.Linear(self.num_prototypes, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.add_on_layers(features)
        
        # Positive cosine similarity (as in original paper)
        features_norm = features / torch.norm(features, dim=1, keepdim=True).clamp(min=1e-6)
        prototypes_norm = self.prototypes / torch.norm(self.prototypes, dim=1, keepdim=True).clamp(min=1e-6)
        similarities = (features_norm @ prototypes_norm.T) * 10.0  # Scaling factor
        
        logits = self.classifier(similarities)
        return logits, similarities, features
    
    def get_class_prototypes(self, class_idx):
        start = class_idx * self.num_prototypes_per_class
        return range(start, start + self.num_prototypes_per_class)

# Loss Functions - Fixed for positive similarity with robust clamps
def compute_prototype_loss(logits, labels, similarities, prototypes, cooccur_matrix, loss_weights):
    classification_loss = nn.BCEWithLogitsLoss()(logits, labels)
    
    N = logits.size(0)
    clustering_loss = 0
    separation_loss = 0
    
    for i in range(N):
        positive_classes = torch.where(labels[i] > 0.5)[0]
        negative_classes = torch.where(labels[i] <= 0.5)[0]
        
        P_pos = []
        for c in positive_classes:
            start = c.item() * MODEL_CONFIG['num_prototypes_per_class']
            end = start + MODEL_CONFIG['num_prototypes_per_class']
            P_pos.extend(range(start, end))
        
        P_neg = []
        for c in negative_classes:
            start = c.item() * MODEL_CONFIG['num_prototypes_per_class']
            end = start + MODEL_CONFIG['num_prototypes_per_class']
            P_neg.extend(range(start, end))
        
        if P_pos:
            clustering_loss += -torch.max(similarities[i, P_pos]).clamp(-10, 10)
        if P_neg:
            separation_loss += torch.max(similarities[i, P_neg]).clamp(-10, 10)
    
    clustering_loss /= N
    separation_loss /= N
    
    # Diversity loss
    P_norm = prototypes / torch.norm(prototypes, dim=1, keepdim=True).clamp(min=1e-6)
    diversity_loss = torch.norm(P_norm @ P_norm.T - torch.eye(prototypes.size(0), device=prototypes.device), 'fro') ** 2
    
    # Contrastive loss with positive similarity and robust clamps
    prototypes_norm = prototypes / torch.norm(prototypes, dim=1, keepdim=True).clamp(min=1e-6)
    proto_sim = (prototypes_norm @ prototypes_norm.T) * 10.0  # Positive similarity
    
    # Expand co-occurrence matrix to match prototype similarities
    num_prototypes_per_class = MODEL_CONFIG['num_prototypes_per_class']
    num_classes = NUM_CLASSES
    
    # Create prototype-to-class mapping
    prototype_classes = []
    for class_idx in range(num_classes):
        prototype_classes.extend([class_idx] * num_prototypes_per_class)
    prototype_classes = torch.tensor(prototype_classes, device=prototypes.device)
    
    # Create expanded co-occurrence matrix for prototypes
    expanded_cooccur = torch.zeros(prototypes.size(0), prototypes.size(0), device=prototypes.device)
    for i in range(prototypes.size(0)):
        for j in range(prototypes.size(0)):
            class_i = prototype_classes[i]
            class_j = prototype_classes[j]
            expanded_cooccur[i, j] = cooccur_matrix[class_i, class_j]
    
    # Compute contrastive loss with robust clamps
    pos_sum = expanded_cooccur.sum().clamp(min=1e-6)
    neg_sum = (1 - expanded_cooccur).sum().clamp(min=1e-6)
    
    cntrst_pos = (expanded_cooccur * proto_sim).sum() / pos_sum
    cntrst_neg = ((1 - expanded_cooccur) * proto_sim).sum() / neg_sum
    contrastive_loss = -(cntrst_pos - cntrst_neg) / np.sqrt(prototypes.size(0))
    
    total_loss = (
        loss_weights['classification'] * classification_loss +
        loss_weights['clustering'] * clustering_loss +
        loss_weights['separation'] * separation_loss +
        loss_weights['diversity'] * diversity_loss +
        loss_weights['contrastive'] * contrastive_loss
    )
    
    return total_loss, {
        'classification': classification_loss.item(),
        'clustering': clustering_loss.item(),
        'separation': separation_loss.item(),
        'diversity': diversity_loss.item(),
        'contrastive': contrastive_loss.item()
    }

# Training with loss component logging and robust AUROC
def train_model(model, train_loader, val_loader, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'], weight_decay=TRAINING_CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    cooccur_path = os.path.join(PREPROCESSING_PATH, 'shd_cooccur.npy')
    if os.path.exists(cooccur_path):
        cooccur = np.load(cooccur_path)
        cooccur_tensor = torch.from_numpy(cooccur).float().to(device)
    else:
        print("Warning: Co-occurrence matrix not found, using identity matrix")
        cooccur_tensor = torch.eye(NUM_CLASSES, device=device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(TRAINING_CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_losses = {'classification': 0, 'clustering': 0, 'separation': 0, 'diversity': 0, 'contrastive': 0}
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits, similarities, features = model(data)
            loss, loss_components = compute_prototype_loss(logits, labels, similarities, model.prototypes, cooccur_tensor, LOSS_WEIGHTS)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['gradient_clip'])
            optimizer.step()
            
            train_loss += loss.item()
            for k, v in loss_components.items():
                train_losses[k] += v
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}: Loss {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_auc = 0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                logits, similarities, features = model(data)
                loss, _ = compute_prototype_loss(logits, labels, similarities, model.prototypes, cooccur_tensor, LOSS_WEIGHTS)
                val_loss += loss.item()
                
                # Collect predictions for AUROC
                val_predictions.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(labels.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate validation AUROC with robust error handling
        if val_predictions:
            val_predictions = np.concatenate(val_predictions)
            val_labels = np.concatenate(val_labels)
            try:
                # Fixed AUROC calculation for multi-label
                val_auc = roc_auc_score(val_labels, val_predictions, average='macro')
            except Exception as e:
                print(f"Warning: Could not calculate AUROC: {e}")
                val_auc = 0.5
        
        # Log loss components
        avg_losses = {k: v / len(train_loader) for k, v in train_losses.items()}
        print(f'Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, Val AUROC {val_auc:.4f}')
        print(f'  Loss components: CE={avg_losses["classification"]:.4f}, Clust={avg_losses["clustering"]:.4f}, '
              f'Sep={avg_losses["separation"]:.4f}, Div={avg_losses["diversity"]:.4f}, Cont={avg_losses["contrastive"]:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= TRAINING_CONFIG['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model

# Evaluation (multi-label)
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            logits, _, _ = model(data)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    sig_logits = 1 / (1 + np.exp(-all_logits))
    
    # Calculate AUROC for each label with robust error handling
    aucs = []
    for i in range(NUM_CLASSES):
        try:
            # Check if we have both positive and negative samples
            if np.sum(all_labels[:, i]) > 0 and np.sum(all_labels[:, i]) < len(all_labels):
                auc = roc_auc_score(all_labels[:, i], sig_logits[:, i])
                aucs.append(auc)
            else:
                print(f"Warning: Label {i} ({SHD_LABEL_COLUMNS[i]}) has no positive or negative samples")
                aucs.append(0.5)
        except Exception as e:
            print(f"Warning: Could not calculate AUROC for label {i} ({SHD_LABEL_COLUMNS[i]}): {e}")
            aucs.append(0.5)
    
    macro_auc = np.mean(aucs)
    print(f"Macro AUROC: {macro_auc:.4f}")
    print("Per-label AUROC:")
    for i, (col, auc) in enumerate(zip(SHD_LABEL_COLUMNS, aucs)):
        print(f"  {col}: {auc:.4f}")
    
    return macro_auc, aucs

# Prototype Projection - Fixed to use cosine similarity and save as .npz
def project_prototypes(model, train_loader, device='cuda'):
    model.eval()
    prototypes = model.prototypes.data
    projected = {}
    
    print("Projecting prototypes using cosine similarity...")
    for p_idx in range(model.num_prototypes):
        max_sim = -1
        projected_ecg = None
        
        for data, _ in train_loader:
            data = data.to(device)
            features = model.add_on_layers(model.feature_extractor(data))
            
            # Use cosine similarity for projection (consistent with training)
            features_norm = features / torch.norm(features, dim=1, keepdim=True).clamp(min=1e-6)
            proto_norm = prototypes[p_idx].unsqueeze(0) / torch.norm(prototypes[p_idx].unsqueeze(0)).clamp(min=1e-6)
            similarities = (features_norm @ proto_norm.T).squeeze()
            
            max_sim_idx = torch.argmax(similarities)
            if similarities[max_sim_idx] > max_sim:
                max_sim = similarities[max_sim_idx]
                projected_ecg = data[max_sim_idx].cpu().numpy()
        
        projected[p_idx] = projected_ecg
        
        if p_idx % 10 == 0:
            print(f"Projected prototype {p_idx}/{model.num_prototypes} (max sim: {max_sim:.4f})")
    
    # Save as .npz for efficiency
    projection_path = os.path.join(OUTPUT_DIR, 'projected_prototypes.npz')
    np.savez(projection_path, **{f'prototype_{i}': projected[i] for i in range(model.num_prototypes)})
    print(f"Saved projected prototypes to {projection_path}")

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'project'], default='train')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        train_loader, val_loader, test_loader = get_echonext_dataloaders(
            batch_size=TRAINING_CONFIG['batch_size'], 
            sampling_rate=DATA_CONFIG['sampling_rate']
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Try setting DATA_CONFIG['use_synthetic_fallback'] = True for testing")
        return
    
    model = ProtoECGNetSHD()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    if args.mode == 'train':
        print("Starting training...")
        model = train_model(model, train_loader, val_loader, device)
        model_path = os.path.join(OUTPUT_DIR, 'final_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Training completed. Model saved to {model_path}")
        
    elif args.mode == 'evaluate':
        model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
            evaluate_model(model, test_loader, device)
        else:
            print(f"Model file not found: {model_path}")
            
    elif args.mode == 'project':
        model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
            project_prototypes(model, train_loader, device)
        else:
            print(f"Model file not found: {model_path}")

if __name__ == '__main__':
    main()
