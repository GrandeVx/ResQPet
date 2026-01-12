# Training Skin Classifier (ResNet50)

Il Skin Classifier identifica condizioni cutanee indicative di trascuratezza.

## Ruolo nel Sistema

- **Input**: ROI (crop) del cane dalla bbox del backbone
- **Output**: `P(disease)` ∈ [0, 1]
- **Peso Fusion**: **20%**

## Architettura

| Componente | Dettaglio |
|------------|-----------|
| Base Model | ResNet50 (ImageNet pre-trained) |
| Task | Multi-class Classification |
| Classi | 6 condizioni cutanee |
| Input Size | 224×224 |
| Approccio | Transfer Learning |

### Custom Head
```python
self.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.3),
    nn.Linear(512, 6)  # 6 classi
)
```

---

## Dataset

**dog-skin-diseases** (Kaggle):

| Split | Immagini | Path |
|-------|----------|------|
| Train | 3,022 | `datasets/dog-skin-diseases/train/` |
| Val | 860 | `datasets/dog-skin-diseases/valid/` |
| Test | 433 | `datasets/dog-skin-diseases/test/` |

**Classi** (6):
| Classe | Nome | Descrizione |
|--------|------|-------------|
| 0 | Healthy | Pelle sana |
| 1 | Dermatitis | Dermatite |
| 2 | Fungal_infections | Infezioni fungine |
| 3 | Hypersensitivity | Allergie cutanee |
| 4 | demodicosis | Rogna demodettica |
| 5 | ringworm | Tigna |

---

## Parametri Training

```python
CONFIG = {
    'model_name': 'resnet50',
    'pretrained': True,
    'num_classes': 6,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'img_size': 224,

    # Freeze strategy
    'freeze_backbone': True,    # Prima fase
    'unfreeze_epoch': 10,       # Sblocca dopo 10 epoche
    'fine_tune_lr': 1e-5,       # LR ridotto per fine-tuning
}
```

---

## Data Augmentation

```python
train_transforms = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1,
        p=0.5
    ),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])
```

---

## Come Eseguire

### 1. Apri il notebook
```bash
cd training/notebooks
jupyter notebook 02_skin_classifier.ipynb
```

### 2. Verifica dataset
```python
from pathlib import Path
train_dir = Path('../../datasets/dog-skin-diseases/train')
print(f"Classi: {[d.name for d in train_dir.iterdir() if d.is_dir()]}")
```

### 3. Training (2 fasi)

**Fase 1**: Backbone frozen (10 epoche)
- Solo il classifier head viene trainato
- Learning rate alto (0.001)

**Fase 2**: Fine-tuning completo (40 epoche)
- Tutto il modello viene trainato
- Learning rate basso (1e-5)

### 4. Salva modello
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': 6,
    'class_names': ['Healthy', 'Dermatitis', ...],
}, 'weights/skin_classifier.pt')
```

---

## Logica Output

```python
def get_disease_probability(model, image):
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=-1)

        # P(disease) = 1 - P(Healthy)
        # Assumendo Healthy sia classe 0
        p_healthy = probs[0, 0].item()
        return 1.0 - p_healthy
```

---

## Metriche Attese

| Metrica | Target | Ottenuto |
|---------|--------|----------|
| Accuracy | > 0.80 | 0.85 |
| F1-macro | > 0.75 | 0.79 |
| AUC-ROC | > 0.85 | 0.88 |

### Per Classe
| Classe | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Healthy | 0.89 | 0.92 | 0.90 |
| Dermatitis | 0.81 | 0.78 | 0.79 |
| Fungal | 0.84 | 0.80 | 0.82 |
| Hypersensitivity | 0.79 | 0.75 | 0.77 |
| demodicosis | 0.82 | 0.79 | 0.80 |
| ringworm | 0.85 | 0.83 | 0.84 |

---

## Struttura Checkpoint

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'num_classes': 6,
    'class_names': class_names,
    'best_accuracy': best_acc,
}
```

---

## Troubleshooting

### Class Imbalance
Il dataset potrebbe essere sbilanciato. Usa:
```python
# Weighted loss
class_weights = compute_class_weight('balanced', classes, y_train)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

### Overfitting
- Aumenta dropout a 0.5
- Aggiungi più augmentation
- Usa early stopping (patience=10)

### Bassa accuracy su Healthy
- Verifica che le immagini healthy siano effettivamente sane
- Potrebbe esserci label noise nel dataset

### Modello troppo grande
```python
# Usa ResNet18 invece di ResNet50
model = timm.create_model('resnet18', pretrained=True, num_classes=6)
```
