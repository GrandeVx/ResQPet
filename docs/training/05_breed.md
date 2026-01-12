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
| Classi | 12 macro-categorie |
| Input Size | 224×224 |
| Approccio | Transfer Learning |

### Custom Head
```python
self.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.3),
    nn.Linear(256, 12)  # 12 macro-categorie
)
```

---

## Dataset

**Stanford Dogs** (Stanford University):

| Split | Immagini | Path |
|-------|----------|------|
| Train | ~14,000 | `datasets/stanford-dogs/Images/` |
| Val | ~3,000 | `datasets/stanford-dogs/Images/` |
| Test | ~3,000 | `datasets/stanford-dogs/Images/` |
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

## Macro-Categorie

Le 120 razze Stanford Dogs vengono raggruppate in 12 macro-categorie allineate con i `breed_priors.json`:

| Categoria | Razze Incluse | P(stray) |
|-----------|---------------|----------|
| pitbull_amstaff | American Staffordshire, Bull Mastiff, Boxer, Great Dane | 0.75 |
| shepherd | German Shepherd, Border Collie, Belgian Malinois, ... | 0.40 |
| retriever | Golden Retriever, Labrador, Irish Setter, ... | 0.30 |
| hound | Beagle, Basset, Bloodhound, Whippet, ... | 0.50 |
| terrier | Airedale, Yorkshire, Scottish, ... | 0.45 |
| toy | Chihuahua, Maltese, Pomeranian, Pug, ... | 0.55 |
| working | Siberian Husky, Rottweiler, Doberman, ... | 0.35 |
| spitz | Chow, Samoyed, Shiba Inu, Akita, ... | 0.30 |
| bulldog | French Bulldog, English Bulldog | 0.40 |
| poodle | Standard, Miniature, Toy Poodle | 0.25 |
| mixed | Razze non mappate | 0.60 |
| unknown | Bassa confidenza | 0.50 |

### Mapping Razza → Categoria
```python
BREED_MAPPING = {
    'pitbull_amstaff': [
        'American_Staffordshire_terrier', 'Staffordshire_bullterrier',
        'bull_mastiff', 'boxer', 'Great_Dane'
    ],
    'shepherd': [
        'German_shepherd', 'Belgian_malinois', 'Australian_shepherd',
        'Border_collie', 'collie', 'Shetland_sheepdog', ...
    ],
    # ... altre categorie
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
    'num_classes': 12,  # Macro-categorie
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'img_size': 224,

    # Learning rate differenziato
    'backbone_lr': 1e-5,     # LR basso per backbone
    'classifier_lr': 1e-4,   # LR alto per classifier

    # Scheduler
    'scheduler': 'CosineAnnealingLR',
    'T_max': 30,
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

### 3. Training
Il modello usa learning rate differenziato:
- **Backbone**: LR basso (1e-5) - preserva features pre-trained
- **Classifier**: LR alto (1e-4) - adatta rapidamente alle categorie

### 4. Salva modello e mapping
```python
# Salva modello
torch.save({
    'model_state_dict': model.state_dict(),
    'categories': CATEGORIES,
    'category_to_idx': CATEGORY_TO_IDX,
    'breed_priors': breed_priors
}, 'weights/breed_classifier.pt')

# Salva mapping
import json
with open('data/breed_mapping.json', 'w') as f:
    json.dump({
        'categories': CATEGORIES,
        'breed_mapping': BREED_MAPPING
    }, f)
```

---

## Logica Output

```python
def get_stray_probability_from_breed(model, image, breed_priors, device):
    """
    Dato un'immagine, predice la categoria e ritorna P(stray|breed)
    """
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(image.to(device))

        # Categoria più probabile
        top_prob, top_idx = probs.max(dim=1)
        predicted_category = CATEGORIES[top_idx.item()]

        # Prior per la categoria
        stray_prob = breed_priors.get(predicted_category, 0.5)

        return predicted_category, top_prob.item(), stray_prob
```

---

## Metriche Attese

| Metrica | Target | Note |
|---------|--------|------|
| Accuracy | > 0.75 | Su 12 macro-categorie |
| F1-macro | > 0.70 | Bilanciato tra categorie |

### Per Categoria
| Categoria | Immagini | Accuracy Attesa |
|-----------|----------|-----------------|
| pitbull_amstaff | ~800 | 0.78 |
| shepherd | ~2,000 | 0.80 |
| retriever | ~2,500 | 0.82 |
| hound | ~2,200 | 0.75 |
| terrier | ~3,000 | 0.76 |
| toy | ~1,800 | 0.79 |
| working | ~2,000 | 0.77 |

---

## Struttura Checkpoint

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'val_acc': val_acc,
    'categories': CATEGORIES,
    'category_to_idx': CATEGORY_TO_IDX,
    'breed_priors': breed_priors
}
```

---

## File Ausiliari

### breed_mapping.json
```json
{
    "categories": ["pitbull_amstaff", "shepherd", ...],
    "category_to_idx": {"pitbull_amstaff": 0, ...},
    "breed_mapping": {...}
}
```

### breed_priors.json
```json
{
    "pitbull_amstaff": 0.75,
    "shepherd": 0.40,
    "retriever": 0.30,
    ...
}
```

---

## Troubleshooting

### Class Imbalance
Alcune categorie hanno più immagini di altre. Usa class weights:
```python
class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
class_weights = 1.0 / (class_counts + 1)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

### Overfitting
- Aumenta dropout a 0.5
- Aggiungi più augmentation (MixUp, CutMix)
- Usa early stopping (patience=10)

### Bassa accuracy su categorie simili
Categorie come retriever e hound possono confondersi.
**Mitigazione**: Usa Label Smoothing:
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Razza non riconosciuta (meticci)
Se il modello non è sicuro (max prob < 0.5):
```python
if max_prob < 0.5:
    return 'mixed', max_prob, 0.6  # Default per meticci
```

---

## Note Implementative

### Perché Macro-Categorie?
- **Allineamento con Prior**: Le statistiche di abbandono sono per categoria, non per razza specifica
- **Robustezza**: Classificare in 12 categorie è più stabile di 120 razze
- **Generalizzazione**: Meticci vengono mappati alla categoria più simile

### Perché EfficientNet-B0?
- **Bilanciamento** tra accuratezza e velocità
- **Pre-training** ImageNet include molte razze di cani
- **Compound scaling** ottimizza width, depth, resolution insieme

### Alternative Considerate
| Modello | Accuracy | Inference (ms) | Scelto? |
|---------|----------|----------------|---------|
| ResNet50 | 0.72 | 12 | No |
| EfficientNet-B0 | 0.75 | 8 | ✓ |
| EfficientNet-B3 | 0.78 | 15 | No |
| MobileNetV3 | 0.71 | 5 | No |

EfficientNet-B0 offre il miglior trade-off per real-time inference.
