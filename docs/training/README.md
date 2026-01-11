# Guida al Training dei Modelli ResQPet

Questa guida spiega come riprodurre il training di tutti i modelli del sistema ResQPet.

## Overview Sistema

ResQPet utilizza **5 modelli ML** che collaborano per calcolare lo **Stray Index**:

```
Frame Input
    │
    ▼
┌─────────────────────────────────────────┐
│      YOLO11 Dog-Pose (Backbone)         │
│   Detection cani + 24 keypoints         │
└─────────────────────────────────────────┘
    │
    ├─────────────┬─────────────┬─────────────┐
    ▼             ▼             ▼             ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Collar  │ │  Skin   │ │  Pose   │ │  Breed  │
│ YOLOv8n │ │ResNet50 │ │   MLP   │ │EffNet-B0│
└─────────┘ └─────────┘ └─────────┘ └─────────┘
    │             │             │             │
    ▼             ▼             ▼             ▼
P(no_collar)  P(disease)  P(stray_pose) P(stray|breed)
    │             │             │             │
    └─────────────┴─────────────┴─────────────┘
                        │
                        ▼
              Weighted Fusion (35%/20%/25%/20%)
                        │
                        ▼
                   Stray Index [0,1]
```

---

## Prerequisiti

### Hardware
- **Minimo**: 8 GB RAM, CPU moderno
- **Consigliato**: 16+ GB RAM, GPU NVIDIA (CUDA 11.8+) o Apple Silicon (MPS)
- **Usato per training originale**: NVIDIA RTX 5090

### Software
```bash
# Python 3.10+
python --version

# Crea virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure: venv\Scripts\activate  # Windows

# Installa dipendenze
pip install -r backend/requirements.txt
pip install jupyter notebook
```

### Dipendenze Principali
```
torch>=2.0.0
ultralytics>=8.0.0
timm>=0.9.0
albumentations>=1.3.0
pandas>=2.0.0
scikit-learn>=1.3.0
jupyter>=1.0.0
```

---

## Ordine di Training

**IMPORTANTE**: I modelli devono essere trainati in questo ordine specifico:

| # | Modello | Notebook | Tempo GPU | Dipendenze |
|---|---------|----------|-----------|------------|
| 1 | Backbone | `00a_yolo_dog_pose_training.ipynb` | 2-3 ore | Nessuna |
| 2 | Collar | `01_collar_detector.ipynb` | 1 ora | Nessuna |
| 3 | Keypoints | `00_keypoints_extraction.ipynb` | 30 min | Backbone |
| 4 | Pose | `03_pose_classifier.ipynb` | 15 min | Keypoints |
| 5 | Skin | `02_skin_classifier.ipynb` | 2 ore | Nessuna |
| 6 | Breed | `04_breed_classifier.ipynb` | 3 ore | Nessuna |

**Tempo totale stimato**: ~8-9 ore con GPU

---

## Quick Start

```bash
# 1. Attiva environment
cd /path/to/ResQPet
source venv/bin/activate

# 2. Avvia Jupyter
cd training/notebooks
jupyter notebook

# 3. Esegui i notebook in ordine (00a → 00 → 01 → 02 → 03 → 04)

# 4. Verifica output
ls ../../weights/
# Dovresti vedere: yolo11n-dog-pose.pt, collar_detector.pt, etc.
```

---

## Dataset Utilizzati

| Dataset | Path | Immagini | Uso |
|---------|------|----------|-----|
| dog-pose | `datasets/dog-pose/` | 8,476 | Backbone training |
| dog-with-leash | `datasets/dog-with-leash/` | 152 | Collar detector |
| dog-skin-diseases | `datasets/dog-skin-diseases/` | 4,316 | Skin classifier |
| stanford-dogs | `datasets/stanford-dogs/` | 20,580 | Breed classifier |
| stray-dogs-fyp | `datasets/stray-dogs-fyp/` | 10,152 | Pose classifier (stray) |

---

## Documentazione Dettagliata

Per ogni modello, consulta la guida specifica:

1. **[01_backbone.md](01_backbone.md)** - YOLO11 Dog-Pose (Backbone)
2. **[02_collar.md](02_collar.md)** - Collar Detector (YOLOv8n)
3. **[03_skin.md](03_skin.md)** - Skin Classifier (ResNet50)
4. **[04_pose.md](04_pose.md)** - Pose Classifier (MLP) - **Weak Supervision**
5. **[05_breed.md](05_breed.md)** - Breed Classifier (EfficientNet-B0)

---

## Output Attesi

Dopo il training completo, la cartella `weights/` conterrà:

```
weights/
├── yolo11n-dog-pose.pt      # 6.1 MB - Backbone
├── collar_detector.pt        # 6.0 MB - Collar
├── skin_classifier.pt        # 90 MB  - Skin
├── stray_pose_classifier.pt  # 80 KB  - Pose
├── breed_classifier.pt       # 50 MB  - Breed
└── yolo11n-dog-pose.onnx    # 12 MB  - ONNX export
```

---

## Verifica Pipeline Completa

Dopo aver trainato tutti i modelli:

```bash
cd training/notebooks
jupyter notebook 05_pipeline_demo.ipynb
```

Questo notebook testa la pipeline end-to-end e mostra:
- Detection cani con keypoints
- Calcolo Stray Index per ogni detection
- Breakdown delle 4 componenti (collar, skin, pose, breed)

---

## Troubleshooting

### CUDA Out of Memory
```python
# Riduci batch size nel notebook
batch_size = 8  # invece di 16
```

### MPS (Apple Silicon) Issues
```python
# Forza uso CPU
device = 'cpu'  # invece di 'mps'
```

### Dataset non trovato
```python
# Verifica path
from pathlib import Path
print(Path('../../datasets/dog-pose').exists())  # Deve essere True
```

### Modello non carica
```python
# Verifica file esiste
import torch
checkpoint = torch.load('../../weights/skin_classifier.pt', weights_only=False)
print(checkpoint.keys())
```

---

## Metriche di Riferimento

Metriche ottenute dal training con RTX 5090:

| Modello | Metrica | Valore |
|---------|---------|--------|
| Backbone | mAP@0.5 | 0.89 |
| Collar | mAP@0.5 | 0.51 |
| Skin | Accuracy | 0.85 |
| Pose | AUC-ROC | 0.78 |
| Breed | Top-5 Acc | 0.82 |

---

## Contributo Originale: Weak Supervision

Il **Pose Classifier** utilizza un approccio di **Weak Supervision**:

- **Label automatici** dall'origine del dataset (non annotazione manuale)
- **Stray (1)**: Keypoints estratti dal FYP Dataset (cani randagi)
- **Owned (0)**: Keypoints estratti da Stanford Dogs (cani padronali)

Questo permette di creare un dataset di ~30,000+ pose senza annotazione manuale.

Vedi [04_pose.md](04_pose.md) per dettagli.
