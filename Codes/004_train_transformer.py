"""
Transformer-based Model for Biliary Pancreatitis Recurrence Prediction
======================================================================

This script implements a Transformer architecture for tabular data with:
- Embeddings for categorical variables
- Linear projections for continuous variables
- Multi-head self-attention
- Training on both prospective and retrospective datasets

Architecture inspired by TabTransformer and FT-Transformer papers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("TRANSFORMER MODEL FOR BILIARY PANCREATITIS PREDICTION")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Model and training configuration"""
    
    # Data configuration
    CATEGORICAL_VARIABLES = [
        'sex', 'previous_episodes', 'admitting_specialty', 'diabetes',
        'chronic_pulmonary_disease', 'hypertension', 'atrial_fibrillation',
        'ischemic_heart_disease', 'chronic_kidney_disease', 'hematopoietic_disease',
        'immunosuppressive_medications', 'choledocholithiasis', 'cholangitis', 'ercp'
    ]
    
    CONTINUOUS_VARIABLES = [
        'age', 'bmi', 'wbc', 'neutrophils', 'platelets', 'inr', 'crp',
        'ast', 'alt', 'total_bilirubin', 'conjugated_bilirubin', 'ggt',
        'serum_amylase', 'serum_lipase', 'ldh'
    ]
    
    # Categorical variable cardinalities (max value + 1 for each)
    # These are determined from the data
    CATEGORICAL_CARDINALITIES = {
        'sex': 3,  # values: 1, 2 -> need 3 embeddings (0 padding + 1,2)
        'previous_episodes': 2,  # values: 0, 1
        'admitting_specialty': 5,  # values: 1-4 -> need 5
        'diabetes': 2,
        'chronic_pulmonary_disease': 2,
        'hypertension': 2,
        'atrial_fibrillation': 2,
        'ischemic_heart_disease': 2,
        'chronic_kidney_disease': 2,
        'hematopoietic_disease': 2,
        'immunosuppressive_medications': 2,
        'choledocholithiasis': 4,  # values: 1-3 -> need 4
        'cholangitis': 2,
        'ercp': 6  # values: 1-5 in retro -> need 6 for safety
    }
    
    # Model architecture
    EMBEDDING_DIM = 32  # Dimension for categorical embeddings
    CONTINUOUS_DIM = 32  # Dimension to project continuous variables
    D_MODEL = 64  # Transformer hidden dimension
    N_HEADS = 4  # Number of attention heads
    N_LAYERS = 3  # Number of transformer layers
    D_FF = 256  # Feed-forward dimension
    DROPOUT = 0.1
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.01
    EPOCHS = 100
    PATIENCE = 15  # Early stopping patience
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    OUTPUT_DIR = 'transformer_models'
    
    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


config = Config()
print(f"\nDevice: {config.DEVICE}")
print(f"Number of categorical features: {len(config.CATEGORICAL_VARIABLES)}")
print(f"Number of continuous features: {len(config.CONTINUOUS_VARIABLES)}")
print(f"Total features: {len(config.CATEGORICAL_VARIABLES) + len(config.CONTINUOUS_VARIABLES)}")


# =============================================================================
# DATASET CLASS
# =============================================================================

class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data with categorical and continuous features"""
    
    def __init__(self, data_path, dataset_type='prospective'):
        """
        Args:
            data_path: Path to CSV file
            dataset_type: 'prospective' or 'retrospective'
        """
        self.data = pd.read_csv(data_path)
        self.dataset_type = dataset_type
        
        # Remove identifier columns
        if dataset_type == 'prospective':
            id_cols = ['patient_id']
        else:
            id_cols = ['country', 'admission_year']
        
        # Extract target
        self.targets = self.data['target'].values
        
        # Extract features
        feature_cols = [col for col in self.data.columns 
                       if col not in id_cols + ['target']]
        self.features = self.data[feature_cols]
        
        # Separate categorical and continuous
        self.categorical_data = self.features[config.CATEGORICAL_VARIABLES].values
        self.continuous_data = self.features[config.CONTINUOUS_VARIABLES].values
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'categorical': torch.tensor(self.categorical_data[idx], dtype=torch.long),
            'continuous': torch.tensor(self.continuous_data[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.long)
        }


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class CategoricalEmbedding(nn.Module):
    """Embedding layer for categorical variables"""
    
    def __init__(self, cardinalities, embedding_dim):
        """
        Args:
            cardinalities: List of cardinality for each categorical feature
            embedding_dim: Dimension of embeddings
        """
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in cardinalities
        ])
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_categorical)
        Returns:
            Tensor of shape (batch_size, num_categorical, embedding_dim)
        """
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embedded, dim=1)


class ContinuousProjection(nn.Module):
    """Linear projection for continuous variables"""
    
    def __init__(self, num_continuous, projection_dim):
        """
        Args:
            num_continuous: Number of continuous features
            projection_dim: Dimension to project each feature to
        """
        super().__init__()
        # Each continuous feature gets its own projection layer
        self.projections = nn.ModuleList([
            nn.Linear(1, projection_dim)
            for _ in range(num_continuous)
        ])
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_continuous)
        Returns:
            Tensor of shape (batch_size, num_continuous, projection_dim)
        """
        projected = [proj(x[:, i:i+1]) for i, proj in enumerate(self.projections)]
        return torch.stack(projected, dim=1)


class TransformerLayer(nn.Module):
    """Single Transformer encoder layer"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-head attention with residual
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class TabularTransformer(nn.Module):
    """
    Transformer model for tabular data classification
    
    Architecture:
    1. Embed categorical features
    2. Project continuous features
    3. Combine into sequence
    4. Apply Transformer layers
    5. Aggregate and classify
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Get cardinalities in order
        cardinalities = [config.CATEGORICAL_CARDINALITIES[var] 
                        for var in config.CATEGORICAL_VARIABLES]
        
        # Embedding for categorical features
        self.categorical_embedding = CategoricalEmbedding(
            cardinalities=cardinalities,
            embedding_dim=config.EMBEDDING_DIM
        )
        
        # Projection for continuous features
        self.continuous_projection = ContinuousProjection(
            num_continuous=len(config.CONTINUOUS_VARIABLES),
            projection_dim=config.CONTINUOUS_DIM
        )
        
        # Project embeddings and projections to d_model
        self.cat_to_model = nn.Linear(config.EMBEDDING_DIM, config.D_MODEL)
        self.cont_to_model = nn.Linear(config.CONTINUOUS_DIM, config.D_MODEL)
        
        # Positional encoding (learnable)
        num_features = len(config.CATEGORICAL_VARIABLES) + len(config.CONTINUOUS_VARIABLES)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, num_features, config.D_MODEL) * 0.02
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                d_model=config.D_MODEL,
                n_heads=config.N_HEADS,
                d_ff=config.D_FF,
                dropout=config.DROPOUT
            )
            for _ in range(config.N_LAYERS)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(config.D_MODEL)
        
        self.classifier = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.D_MODEL, 2)  # Binary classification
        )
        
        # Class token for aggregation (like BERT's [CLS])
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.D_MODEL))
        
    def forward(self, categorical, continuous):
        """
        Args:
            categorical: (batch_size, num_categorical)
            continuous: (batch_size, num_continuous)
        Returns:
            logits: (batch_size, 2)
        """
        batch_size = categorical.size(0)
        
        # Embed categorical features
        cat_embedded = self.categorical_embedding(categorical)  # (B, N_cat, emb_dim)
        cat_projected = self.cat_to_model(cat_embedded)  # (B, N_cat, d_model)
        
        # Project continuous features
        cont_projected = self.continuous_projection(continuous)  # (B, N_cont, proj_dim)
        cont_projected = self.cont_to_model(cont_projected)  # (B, N_cont, d_model)
        
        # Concatenate features
        x = torch.cat([cat_projected, cont_projected], dim=1)  # (B, N_cat+N_cont, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+N_features, d_model)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Use CLS token for classification
        x = x[:, 0]  # (B, d_model)
        x = self.norm(x)
        
        # Classification
        logits = self.classifier(x)  # (B, 2)
        
        return logits


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

class Trainer:
    """Training and evaluation handler"""
    
    def __init__(self, model, config, train_loader, val_loader, test_loader):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Calculate class weights for imbalanced data
        train_targets = []
        for batch in train_loader:
            train_targets.extend(batch['target'].numpy())
        train_targets = np.array(train_targets)
        
        class_counts = np.bincount(train_targets)
        class_weights = len(train_targets) / (len(class_counts) * class_counts)
        self.class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)
        
        print(f"\nClass weights: {class_weights}")
        
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_auc = 0
        self.patience_counter = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            categorical = batch['categorical'].to(self.config.DEVICE)
            continuous = batch['continuous'].to(self.config.DEVICE)
            targets = batch['target'].to(self.config.DEVICE)
            
            # Forward pass
            logits = self.model(categorical, continuous)
            loss = self.criterion(logits, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, loader):
        """Evaluate on a dataset"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                categorical = batch['categorical'].to(self.config.DEVICE)
                continuous = batch['continuous'].to(self.config.DEVICE)
                targets = batch['target'].to(self.config.DEVICE)
                
                logits = self.model(categorical, continuous)
                loss = self.criterion(logits, targets)
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        avg_loss = total_loss / len(loader)
        auc = roc_auc_score(all_targets, all_probs)
        auprc = average_precision_score(all_targets, all_probs)
        acc = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='binary', zero_division=0
        )
        
        metrics = {
            'loss': avg_loss,
            'auc': auc,
            'auprc': auprc,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def train(self):
        """Full training loop"""
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        for epoch in range(self.config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.EPOCHS}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            val_metrics = self.evaluate(self.val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}")
            print(f"Val AUPRC: {val_metrics['auprc']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['auc'])
            
            # Save best model
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                print(f"✓ New best model! (AUC: {self.best_val_auc:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model for final evaluation
        self.load_checkpoint('best_model.pt')
        
        # Final evaluation on test set
        print("\n" + "=" * 80)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 80)
        
        test_metrics = self.evaluate(self.test_loader)
        
        print(f"\nTest AUC: {test_metrics['auc']:.4f}")
        print(f"Test AUPRC: {test_metrics['auprc']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        
        return test_metrics
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        path = os.path.join(self.config.OUTPUT_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_auc': self.best_val_auc
        }, path)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        path = os.path.join(self.config.OUTPUT_DIR, filename)
        checkpoint = torch.load(path, map_location=self.config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_auc = checkpoint['best_val_auc']
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUC
        axes[0, 1].plot(self.history['val_auc'], label='Validation AUC')
        axes[0, 1].axhline(y=self.best_val_auc, color='r', linestyle='--', 
                          label=f'Best: {self.best_val_auc:.4f}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_title('Validation AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy
        axes[1, 0].plot(self.history['val_acc'], label='Validation Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.history['learning_rate'], label='Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.OUTPUT_DIR, 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Training plots saved to: {plot_path}")
        plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Choose dataset to train on
    dataset_choice = input("\nWhich dataset to train on?\n1. Prospective\n2. Retrospective\n3. Both (combined)\nEnter choice (1/2/3): ").strip()
    
    if dataset_choice == '1':
        print("\n✓ Training on Prospective dataset")
        train_dataset = TabularDataset('processed_data/prospective/train.csv', 'prospective')
        val_dataset = TabularDataset('processed_data/prospective/val.csv', 'prospective')
        test_dataset = TabularDataset('processed_data/prospective/test.csv', 'prospective')
        model_suffix = 'prospective'
        
    elif dataset_choice == '2':
        print("\n✓ Training on Retrospective dataset")
        train_dataset = TabularDataset('processed_data/retrospective/train.csv', 'retrospective')
        val_dataset = TabularDataset('processed_data/retrospective/val.csv', 'retrospective')
        test_dataset = TabularDataset('processed_data/retrospective/test.csv', 'retrospective')
        model_suffix = 'retrospective'
        
    elif dataset_choice == '3':
        print("\n✓ Training on Combined dataset")
        # Load both datasets
        prosp_train = TabularDataset('processed_data/prospective/train.csv', 'prospective')
        prosp_val = TabularDataset('processed_data/prospective/val.csv', 'prospective')
        prosp_test = TabularDataset('processed_data/prospective/test.csv', 'prospective')
        
        retro_train = TabularDataset('processed_data/retrospective/train.csv', 'retrospective')
        retro_val = TabularDataset('processed_data/retrospective/val.csv', 'retrospective')
        retro_test = TabularDataset('processed_data/retrospective/test.csv', 'retrospective')
        
        # Combine datasets
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([prosp_train, retro_train])
        val_dataset = ConcatDataset([prosp_val, retro_val])
        test_dataset = ConcatDataset([prosp_test, retro_test])
        model_suffix = 'combined'
        
    else:
        print("Invalid choice. Using Prospective by default.")
        train_dataset = TabularDataset('processed_data/prospective/train.csv', 'prospective')
        val_dataset = TabularDataset('processed_data/prospective/val.csv', 'prospective')
        test_dataset = TabularDataset('processed_data/prospective/test.csv', 'prospective')
        model_suffix = 'prospective'
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)
    
    model = TabularTransformer(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nModel architecture:")
    print(model)
    
    # Create trainer
    trainer = Trainer(model, config, train_loader, val_loader, test_loader)
    
    # Train model
    test_metrics = trainer.train()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save final results
    results = {
        'model_suffix': model_suffix,
        'config': {
            'embedding_dim': config.EMBEDDING_DIM,
            'continuous_dim': config.CONTINUOUS_DIM,
            'd_model': config.D_MODEL,
            'n_heads': config.N_HEADS,
            'n_layers': config.N_LAYERS,
            'd_ff': config.D_FF,
            'dropout': config.DROPOUT,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY
        },
        'test_metrics': test_metrics,
        'best_val_auc': trainer.best_val_auc,
        'total_epochs': len(trainer.history['train_loss'])
    }
    
    results_path = os.path.join(config.OUTPUT_DIR, f'results_{model_suffix}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nBest Validation AUC: {trainer.best_val_auc:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"\nModel saved to: {config.OUTPUT_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
