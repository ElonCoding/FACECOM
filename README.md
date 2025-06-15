# FaceCom Multi-task Learning Project

This project implements a multi-task learning solution for face analysis, combining gender classification and face recognition tasks using a shared CNN backbone.

## Project Structure

```
├── src/
│   ├── dataset.py      # Dataset and data loading utilities
│   ├── model.py        # Neural network architecture
│   ├── trainer.py      # Training and evaluation logic
│   ├── main.py         # Main training pipeline
│   └── run.py          # Entry point script
├── data/               # Dataset directory
│   └── facecom/        # FaceCom dataset
├── checkpoints/        # Model checkpoints
└── README.md          # Project documentation
```

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- albumentations
- scikit-learn
- pandas
- numpy
- matplotlib
- tqdm

Install dependencies:

```bash
pip install torch torchvision albumentations scikit-learn pandas numpy matplotlib tqdm
```

## Dataset Structure

Place your FaceCom dataset in the `data/facecom` directory with the following structure:

```
data/facecom/
├── images/            # Image files
└── annotations.json   # Annotation file
```

## Training

1. Update the configuration in `src/main.py` if needed:
   - Adjust learning rate, batch size, epochs
   - Modify model architecture parameters
   - Change loss weights (alpha for gender, beta for identity)

2. Run the training:

```bash
python src/run.py
```

Optional arguments:
```bash
python src/run.py --data_dir path/to/data --batch_size 32 --epochs 30 --lr 0.001 --seed 42
```

## Model Architecture

- Backbone: ResNet-50 (pretrained)
- Two task-specific heads:
  - Gender Classification Head (Binary)
  - Face Recognition Head (Multi-class)

## Features

- Multi-task learning with shared backbone
- Comprehensive data augmentation for challenging visual conditions
- Early stopping and model checkpointing
- Detailed metric tracking and visualization
- Production-ready code structure

## Metrics

The model tracks the following metrics:

### Gender Classification
- Accuracy
- Precision
- Recall
- F1 Score

### Face Recognition
- Top-1 Accuracy
- Macro F1 Score

### Combined Score
- Weighted combination (30% Gender, 70% Identity)

## Output

- Training metrics are plotted and saved as `training_metrics.png`
- Best model is saved in the `checkpoints` directory
- Predictions are saved in `submission.csv`

## License

