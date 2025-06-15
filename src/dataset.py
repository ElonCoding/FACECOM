import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FaceComDataset(Dataset):
    def __init__(self, data_dir, annotation_file, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        # Create label mappings
        self.gender_to_idx = {'Male': 0, 'Female': 1}
        self.identity_to_idx = {}
        identity_set = set()
        for item in self.annotations:
            identity_set.add(item['person_id'])
        self.identity_to_idx = {id_: idx for idx, id_ in enumerate(sorted(identity_set))}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        img_path = os.path.join(self.data_dir, item['filename'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((224, 224))  # Resize to match model input size
        
        # Apply transformations
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        
        # Get labels
        gender_label = self.gender_to_idx[item['gender']]
        identity_label = self.identity_to_idx[item['person_id']]
        
        return {
            'image': image,
            'gender': torch.tensor(gender_label, dtype=torch.long),
            'identity': torch.tensor(identity_label, dtype=torch.long)
        }

def get_transforms(mode):
    if mode == 'train':
        return A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomRain(p=0.2),
            A.RandomFog(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def create_dataloaders(config):
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    train_dataset = FaceComDataset(
        data_dir=config['data_dir'],
        annotation_file=config['annotation_file'],
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = FaceComDataset(
        data_dir=config['data_dir'],
        annotation_file=config['annotation_file'],
        transform=val_transform,
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader