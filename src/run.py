import argparse
from main import main

def parse_args():
    parser = argparse.ArgumentParser(description='Train FaceCom Multi-task Model')
    
    # Add arguments for customization if needed
    parser.add_argument('--data_dir', type=str, default='data/facecom',
                        help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main()