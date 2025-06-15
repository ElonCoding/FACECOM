import os
import random
import numpy as np
import torch
from dataset import create_dataloaders
from model import create_model
from trainer import Trainer
import matplotlib.pyplot as plt

# Set seed for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration
CONFIG = {
    'data_dir': 'data/facecom',  # Update with your data directory
    'annotation_file': 'data/facecom/annotations.json',  # Update with your annotation file
    'img_size': 224,
    'batch_size': 32,
    'num_workers': 4,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'epochs': 50,
    'early_stopping_patience': 10,
    'alpha': 0.5,  # Weight for gender loss
    'beta': 0.5,   # Weight for identity loss
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'seed': 42,
    'checkpoint_dir': 'checkpoints',
    'submission_file': 'submission.csv'
}

def plot_metrics(metrics_history):
    plt.figure(figsize=(15, 5))
    
    # Plot gender metrics
    plt.subplot(1, 2, 1)
    for metric in ['gender_accuracy', 'gender_precision', 'gender_recall', 'gender_f1']:
        plt.plot(metrics_history[metric], label=metric)
    plt.title('Gender Classification Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Plot identity metrics
    plt.subplot(1, 2, 2)
    for metric in ['identity_accuracy', 'identity_f1']:
        plt.plot(metrics_history[metric], label=metric)
    plt.title('Identity Recognition Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def main():
    # Set random seed
    seed_everything(CONFIG['seed'])
    
    # Create checkpoint directory
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(CONFIG)
    
    # Get number of identities from the dataset
    num_identities = len(train_loader.dataset.identity_to_idx)
    
    # Create model
    model = create_model(CONFIG, num_identities)
    
    # Create trainer
    trainer = Trainer(model, CONFIG)
    
    # Training loop
    metrics_history = {
        'gender_accuracy': [], 'gender_precision': [],
        'gender_recall': [], 'gender_f1': [],
        'identity_accuracy': [], 'identity_f1': []
    }
    
    best_combined_score = 0
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        print(f'\nEpoch {epoch+1}/{CONFIG["epochs"]}')
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics, combined_score = trainer.validate(val_loader)
        
        # Update metrics history
        for metric in metrics_history:
            metrics_history[metric].append(val_metrics[metric])
        
        # Print metrics
        print('\nValidation Metrics:')
        print(f'Gender - Accuracy: {val_metrics["gender_accuracy"]:.4f}, '
              f'F1: {val_metrics["gender_f1"]:.4f}')
        print(f'Identity - Accuracy: {val_metrics["identity_accuracy"]:.4f}, '
              f'F1: {val_metrics["identity_f1"]:.4f}')
        print(f'Combined Score: {combined_score:.4f}')
        
        # Plot metrics
        plot_metrics(metrics_history)
        
        # Early stopping
        if trainer.patience_counter >= CONFIG['early_stopping_patience']:
            print('\nEarly stopping triggered!')
            break
    
    # Load best model and create submission
    trainer.load_checkpoint('best_model.pth')
    trainer.create_submission(val_loader, CONFIG['submission_file'])
    print(f'\nSubmission file created: {CONFIG["submission_file"]}')

if __name__ == '__main__':
    main()