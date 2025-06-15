import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config['device']
        
        # Loss functions
        self.gender_criterion = nn.BCEWithLogitsLoss()
        self.identity_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=self.config['early_stopping_patience'] // 2, factor=0.5
        )
        
        # Best metrics for model saving
        self.best_val_score = 0
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_losses = []
        gender_preds, gender_labels = [], []
        identity_preds, identity_labels = [], []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            images = batch['image'].to(self.device)
            gender_label = batch['gender'].to(self.device)
            identity_label = batch['identity'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate losses
            gender_loss = self.gender_criterion(outputs['gender'], gender_label.float())
            identity_loss = self.identity_criterion(outputs['identity'], identity_label)
            
            # Combined loss
            total_loss = (self.config['alpha'] * gender_loss + 
                         self.config['beta'] * identity_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Store predictions and labels
            gender_preds.extend((torch.sigmoid(outputs['gender']) > 0.5).cpu().numpy())
            gender_labels.extend(gender_label.cpu().numpy())
            identity_preds.extend(torch.argmax(outputs['identity'], dim=1).cpu().numpy())
            identity_labels.extend(identity_label.cpu().numpy())
            
            epoch_losses.append(total_loss.item())
            pbar.set_postfix({'loss': np.mean(epoch_losses[-100:])})  # Show running average loss
        
        # Calculate metrics
        metrics = self.calculate_metrics(gender_preds, gender_labels,
                                       identity_preds, identity_labels)
        return metrics
    
    def validate(self, val_loader):
        self.model.eval()
        val_losses = []
        gender_preds, gender_labels = [], []
        identity_preds, identity_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                gender_label = batch['gender'].to(self.device)
                identity_label = batch['identity'].to(self.device)
                
                outputs = self.model(images)
                
                # Calculate losses
                gender_loss = self.gender_criterion(outputs['gender'], gender_label.float())
                identity_loss = self.identity_criterion(outputs['identity'], identity_label)
                total_loss = (self.config['alpha'] * gender_loss + 
                             self.config['beta'] * identity_loss)
                
                # Store predictions and labels
                gender_preds.extend((torch.sigmoid(outputs['gender']) > 0.5).cpu().numpy())
                gender_labels.extend(gender_label.cpu().numpy())
                identity_preds.extend(torch.argmax(outputs['identity'], dim=1).cpu().numpy())
                identity_labels.extend(identity_label.cpu().numpy())
                
                val_losses.append(total_loss.item())
        
        # Calculate metrics
        metrics = self.calculate_metrics(gender_preds, gender_labels,
                                       identity_preds, identity_labels)
        
        # Calculate combined score (30% gender, 70% identity)
        combined_score = (0.3 * metrics['gender_accuracy'] +
                         0.7 * metrics['identity_accuracy'])
        
        # Early stopping check
        if combined_score > self.best_val_score:
            self.best_val_score = combined_score
            self.patience_counter = 0
            self.save_checkpoint('best_model.pth')
        else:
            self.patience_counter += 1
        
        self.scheduler.step(combined_score)
        
        return metrics, combined_score
    
    def calculate_metrics(self, gender_preds, gender_labels, identity_preds, identity_labels):
        # Gender metrics
        gender_accuracy = accuracy_score(gender_labels, gender_preds)
        gender_precision = precision_score(gender_labels, gender_preds, average='binary')
        gender_recall = recall_score(gender_labels, gender_preds, average='binary')
        gender_f1 = f1_score(gender_labels, gender_preds, average='binary')
        
        # Identity metrics
        identity_accuracy = accuracy_score(identity_labels, identity_preds)
        identity_f1 = f1_score(identity_labels, identity_preds, average='macro')
        
        return {
            'gender_accuracy': gender_accuracy,
            'gender_precision': gender_precision,
            'gender_recall': gender_recall,
            'gender_f1': gender_f1,
            'identity_accuracy': identity_accuracy,
            'identity_f1': identity_f1
        }
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_score': self.best_val_score,
            'config': self.config
        }
        torch.save(checkpoint, os.path.join(self.config['checkpoint_dir'], filename))
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(os.path.join(self.config['checkpoint_dir'], filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_score = checkpoint['best_val_score']
    
    def create_submission(self, test_loader, submission_file):
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Creating submission'):
                images = batch['image'].to(self.device)
                outputs = self.model(images)
                
                gender_preds = (torch.sigmoid(outputs['gender']) > 0.5).cpu().numpy()
                identity_preds = torch.argmax(outputs['identity'], dim=1).cpu().numpy()
                
                for g, i in zip(gender_preds, identity_preds):
                    predictions.append({
                        'predicted_gender': 'Male' if g == 0 else 'Female',
                        'predicted_identity': int(i)
                    })
        
        # Create submission DataFrame
        df = pd.DataFrame(predictions)
        df.to_csv(submission_file, index=False)