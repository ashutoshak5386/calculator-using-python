## Early Detection of Alzheimer’s Disease Using AI (Starter Project)

This project provides a minimal, runnable PyTorch pipeline to fine-tune an image classifier (ResNet18) for early Alzheimer’s detection from MRI or PET scan images organized in folders.

### 1) Features
- Transfer learning with ResNet18
- ImageFolder loaders with standard medical-imaging friendly augmentations
- Early stopping and model checkpointing
- Evaluation with accuracy, precision, recall, F1, and confusion matrix plot
- Simple inference for a single image or an entire folder

### 2) Folder Structure
```
.
├── alz_ai/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

### 3) Expected Data Layout
Use `torchvision.datasets.ImageFolder`-style directories:
```
DATA_ROOT/
├── train/
│   ├── ClassA/
│   │   ├── img1.png
│   │   └── ...
│   └── ClassB/
│       ├── img2.png
│       └── ...
├── val/
│   ├── ClassA/
│   └── ClassB/
└── test/               # optional
    ├── ClassA/
    └── ClassB/
```

You can name classes anything (e.g., `CN`, `MCI`, `AD`). The class names are saved during training and reused for evaluation/inference.

### 4) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5) Training
```bash
python -m alz_ai.train \
  --data-dir /path/to/DATA_ROOT \
  --output-dir runs/exp1 \
  --epochs 15 \
  --batch-size 32 \
  --lr 3e-4 \
  --img-size 224 \
  --patience 5
```

Artifacts saved to `--output-dir`:
- `best.pt`: best checkpoint by validation loss
- `last.pt`: last checkpoint
- `classes.txt`: class names in the model’s index order
- `confusion_matrix.png` (after evaluation)

### 6) Evaluation
Evaluate on `val` or `test` split:
```bash
python -m alz_ai.evaluate \
  --data-dir /path/to/DATA_ROOT \
  --split val \
  --checkpoint runs/exp1/best.pt \
  --output-dir runs/exp1
```

### 7) Inference
Single image:
```bash
python -m alz_ai.infer \
  --image /path/to/image.png \
  --checkpoint runs/exp1/best.pt
```

Folder of images:
```bash
python -m alz_ai.infer \
  --folder /path/to/images \
  --checkpoint runs/exp1/best.pt
```

### 8) Notes
- This is a baseline for experimentation. For clinical use, rigorous validation and regulatory considerations are required.
- Consider domain-specific preprocessing (e.g., skull-stripping, intensity normalization) before training.
- You can swap the backbone in `alz_ai/model.py` to larger models like ResNet50 or ViT if you have more data and compute.
# calculator-using-python