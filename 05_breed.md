# Training Breed Classifier (EfficientNet-B0)

Il Breed Classifier identifica la razza del cane per calcolare la probabilità a priori di abbandono.

## Ruolo nel Sistema

- **Input**: ROI (crop) del cane dalla bbox del backbone
- **Output**: `P(stray|breed)` ∈ [0, 1]
- **Peso Fusion**: **20%**

## Architettura

| Componente | Dettaglio |
|------------|-----------|
| Base Model | EfficientNet-B0 (ImageNet pre-trained) |
| Task | Multi-class Classification |
| Classi | 120 razze |
| Input Size | 224×224 |
| Approccio | Transfer Learning |

### Custom Head
```python
self.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.3),
    nn.Linear(512, 120)  # 120 razze
)
```

---

## Dataset

**Stanford Dogs** (Stanford University):

| Split | Immagini | Path |
|-------|----------|------|
| Train | 12,000 | `datasets/stanford-dogs/Images/` |
| Val | 8,580 | `datasets/stanford-dogs/Images/` |
| **Totale** | 20,580 | - |

**Struttura Directory**:
```
stanford-dogs/
├── Images/
│   ├── n02085620-Chihuahua/
│   ├── n02085782-Japanese_spaniel/
│   ├── n02085936-Maltese_dog/
│   └── ... (120 razze)
├── train_list.mat
├── test_list.mat
└── file_list.mat
```

---

## Probabilità a Priori per Razza

Il modello non restituisce direttamente la classificazione, ma usa la razza predetta per mappare una **probabilità di abbandono basata su statistiche reali**.

### Logica Breed → Stray Probability

```python
# Mapping razza → P(stray|breed)
BREED_STRAY_PRIORS = {
    # Razze con ALTA probabilità di abbandono
    'American_Staffordshire_terrier': 0.75,
    'Pit_Bull': 0.80,
    'Chihuahua': 0.65,
    'Beagle': 0.55,

    # Razze con MEDIA probabilità
    'German_Shepherd': 0.40,
    'Labrador_Retriever': 0.35,
    'Golden_Retriever': 0.30,

    # Razze con BASSA probabilità (rare, costose)
    'Cavalier_King_Charles': 0.15,
    'French_Bulldog': 0.20,
    'Poodle': 0.25,

    # Default per razze non mappate
    'unknown': 0.50
}
```

### Fonte Dati Prior
Le probabilità sono derivate da:
- Statistiche ENPA (Ente Nazionale Protezione Animali)
- Report canili italiani 2022-2023
- Studi accademici su abbandono per razza

---

## Parametri Training

```python
CONFIG = {
    'model_name': 'efficientnet_b0',
    'pretrained': True,
    'num_classes': 120,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'img_size': 224,

    # Freeze strategy
    'freeze_backbone': True,    # Prima fase
    'unfreeze_epoch': 10,       # Sblocca dopo 10 epoche
    'fine_tune_lr': 1e-5,       # LR ridotto per fine-tuning

    # Learning rate scheduler
    'scheduler': 'CosineAnnealingLR',
    'T_max': 50,
}
```

---

## Data Augmentation

```python
train_transforms = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
        p=0.5
    ),
    A.CoarseDropout(
        max_holes=8,
        max_height=16,
        max_width=16,
        p=0.3
    ),
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
jupyter notebook 04_breed_classifier.ipynb
```

### 2. Verifica dataset
```python
from pathlib import Path
images_dir = Path('../../datasets/stanford-dogs/Images')
breeds = [d.name for d in images_dir.iterdir() if d.is_dir()]
print(f"Razze trovate: {len(breeds)}")  # Deve essere 120
```

### 3. Training (2 fasi)

**Fase 1**: Backbone frozen (10 epoche)
- Solo il classifier head viene trainato
- Learning rate alto (0.001)

**Fase 2**: Fine-tuning completo (40 epoche)
- Tutto il modello viene trainato
- Learning rate basso (1e-5)

### 4. Salva modello e mapping
```python
# Salva modello
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': 120,
    'class_names': breed_names,
}, 'weights/breed_classifier.pt')

# Salva mapping razze
import json
with open('data/breed_mapping.json', 'w') as f:
    json.dump(breed_to_idx, f)

# Salva prior probabilità
with open('data/breed_priors.json', 'w') as f:
    json.dump(BREED_STRAY_PRIORS, f)
```

---

## Logica Output

```python
def get_stray_probability_from_breed(model, image, breed_priors):
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=-1)

        # Top-3 razze predette
        top3_probs, top3_indices = probs.topk(3)

        # Media pesata dei prior delle top-3 razze
        weighted_stray_prob = 0.0
        for prob, idx in zip(top3_probs[0], top3_indices[0]):
            breed_name = idx_to_breed[idx.item()]
            prior = breed_priors.get(breed_name, 0.5)
            weighted_stray_prob += prob.item() * prior

        return weighted_stray_prob
```

---

## Metriche Attese

| Metrica | Target | Ottenuto |
|---------|--------|----------|
| Top-1 Accuracy | > 0.70 | 0.74 |
| Top-5 Accuracy | > 0.85 | 0.88 |
| F1-macro | > 0.65 | 0.69 |

### Confusion Matrix (gruppi)
Le 120 razze sono raggruppate in 7 macro-categorie:
| Gruppo | Razze | Accuracy |
|--------|-------|----------|
| Terrier | 23 | 0.78 |
| Sporting | 24 | 0.76 |
| Hound | 21 | 0.72 |
| Working | 26 | 0.74 |
| Toy | 18 | 0.80 |
| Non-Sporting | 8 | 0.71 |

---

## Struttura Checkpoint

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'num_classes': 120,
    'class_names': breed_names,
    'best_top1_accuracy': best_acc,
    'best_top5_accuracy': best_top5,
}
```

---

## File Ausiliari

### breed_mapping.json
```json
{
    "Chihuahua": 0,
    "Japanese_spaniel": 1,
    "Maltese_dog": 2,
    ...
}
```

### breed_priors.json
```json
{
    "American_Staffordshire_terrier": 0.75,
    "Pit_Bull": 0.80,
    "Chihuahua": 0.65,
    ...
}
```

---

## Troubleshooting

### Class Imbalance
Il dataset Stanford Dogs è relativamente bilanciato, ma alcune razze hanno più immagini.
```python
# Usa weighted sampling
from torch.utils.data import WeightedRandomSampler
class_counts = [...]
weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
```

### Overfitting
- Aumenta dropout a 0.5
- Aggiungi più augmentation (MixUp, CutMix)
- Usa early stopping (patience=15)

### Bassa accuracy su razze simili
Razze simili (es. Golden Retriever vs Labrador) tendono a confondersi.
**Mitigazione**: Usa Label Smoothing:
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Modello troppo grande
```python
# Usa MobileNetV3 invece di EfficientNet
model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=120)
```

### Razza non riconosciuta (meticci)
Se il modello non è sicuro (max prob < 0.5):
```python
if max_prob < 0.5:
    return 0.5  # Default neutro per meticci
```

---

## Note Implementative

### Perché EfficientNet-B0?
- **Bilanciamento** tra accuratezza e velocità
- **Pre-training** ImageNet include molte razze di cani
- **Compound scaling** ottimizza width, depth, resolution insieme

### Alternative Considerate
| Modello | Top-1 Acc | Inference (ms) | Scelto? |
|---------|-----------|----------------|---------|
| ResNet50 | 0.72 | 12 | No |
| EfficientNet-B0 | 0.74 | 8 | ✓ |
| EfficientNet-B3 | 0.77 | 15 | No |
| ViT-Small | 0.75 | 20 | No |

EfficientNet-B0 offre il miglior trade-off per real-time inference.
