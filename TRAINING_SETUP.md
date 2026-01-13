# ResQPet - Training Setup Guide

Questa guida descrive come configurare l'ambiente per il training dei modelli ResQPet.

## Download File Precompilati

### Weights (Modelli Trainati)
I pesi dei modelli già trainati sono disponibili su Google Drive:

**[Download weights.zip (~150MB)](https://drive.google.com/file/d/1wTCRK-MADDg0DvcIja4IO2pBX-KKIM0F/view?usp=sharing)**

Contenuto:
- `skin_classifier.pt` - ResNet50 per patologie cutanee (Acc: 0.806)
- `breed_classifier.pt` - EfficientNet-B0 per razze (Acc: 0.863)
- `stray_pose_classifier.pt` - MLP per postura (AUC: 0.633)
- `collar_detector.pt` - YOLOv8n per collare (mAP: 0.853)
- `yolo11n-dog-pose.pt` - Backbone YOLO11 Dog-Pose (mAP: 0.987)

### Training Data (File Derivati)
I file derivati dal preprocessing sono disponibili su Google Drive:

**[Download training_data.zip (~3MB)](https://drive.google.com/file/d/1WTwwJ0ykU3MprUSsO5lvd0oEVDcRt1gv/view?usp=sharing)**

Contenuto:
- `data/keypoints/` - Keypoints estratti per Pose Classifier
- `data/breed_mapping.json` - Mapping razze → categorie
- `data/breed_priors.json` - Prior P(stray|breed)

---

## Setup da Zero (Training Completo)

Se vuoi ri-trainare i modelli da zero, segui questi passi.

### 1. Struttura Directory

```
ResQPet/
├── weights/              # Modelli trainati (output)
├── data/                 # File derivati (output)
└── datasets/             # Dataset originali (da scaricare)
    ├── Stanford Dog/
    ├── Dog's skin diseases/
    └── stray-dogs-fyp/
```

### 2. Download Dataset Originali

#### Stanford Dogs Dataset (Breed Classifier)
- **Fonte**: [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)
- **Dimensione**: ~800MB
- **Struttura**: `Images/n02085620-Chihuahua/`, `Images/n02085782-Japanese_spaniel/`, ...
- **Posizione**: Estrarre in `../Stanford Dog/` (stesso livello di ResQPet)

```bash
# Da Kaggle CLI
kaggle datasets download -d jessicali9530/stanford-dogs-dataset
unzip stanford-dogs-dataset.zip -d "../Stanford Dog"
```

#### Dog's Skin Diseases (Skin Classifier)
- **Fonte**: [Kaggle](https://www.kaggle.com/datasets/deepakdewani/dog-skin-disease-dataset)
- **Dimensione**: ~500MB
- **Classi**: Healthy, Dermatitis, Fungal_infections, Hypersensitivity, demodicosis, ringworm
- **Posizione**: Estrarre in `../Dog's skin diseases/`

```bash
# Da Kaggle CLI
kaggle datasets download -d deepakdewani/dog-skin-disease-dataset
unzip dog-skin-disease-dataset.zip -d "../Dog's skin diseases"
```

#### FYP Stray Dogs Dataset (Pose Classifier - Weak Labels)
- **Fonte**: [Kaggle](https://www.kaggle.com/datasets/faizanurrahman/stray-dogs-detection)
- **Dimensione**: ~1GB
- **Utilizzo**: Keypoints con label "stray" per weak supervision
- **Posizione**: Estrarre in `../stray-dogs-fyp/`

```bash
kaggle datasets download -d faizanurrahman/stray-dogs-detection
unzip stray-dogs-detection.zip -d "../stray-dogs-fyp"
```

#### Dog-Pose Keypoints (YOLO11 Dog-Pose Training)
- **Fonte**: [Universe Roboflow](https://universe.roboflow.com/animal-pose/dog-pose-cxz9o)
- **Dimensione**: ~400MB
- **Annotazioni**: 24 keypoints anatomici del cane
- **Posizione**: `datasets/dog-pose/`

### 3. Ordine di Esecuzione Notebook

```bash
# 1. Training backbone dog-pose (se non usi weights pretrainati)
jupyter notebook training/notebooks/00a_yolo_dog_pose_training.ipynb

# 2. Estrazione keypoints dai dataset
jupyter notebook training/notebooks/00_keypoints_extraction.ipynb

# 3. Training classificatori (in parallelo o sequenza)
jupyter notebook training/notebooks/02_skin_classifier.ipynb
jupyter notebook training/notebooks/03_pose_classifier.ipynb
jupyter notebook training/notebooks/04_breed_classifier.ipynb

# 4. Test pipeline completa
jupyter notebook training/notebooks/05_pipeline_demo.ipynb
```

### 4. Requisiti Hardware

| Configurazione | GPU | Tempo Training |
|----------------|-----|----------------|
| Minima | 1x GTX 1080 (8GB) | ~8 ore |
| Raccomandata | 1x RTX 3090 (24GB) | ~3 ore |
| Ottimale | 2x RTX 5090 (32GB) | ~1 ora |

I notebook supportano automaticamente multi-GPU con DataParallel.

### 5. Requisiti Software

```bash
pip install torch torchvision timm ultralytics
pip install albumentations opencv-python pillow
pip install scikit-learn pandas numpy matplotlib seaborn
pip install tqdm jupyter
```

---

## Note Importanti

### Weak Supervision (Pose Classifier)
Il Pose Classifier usa **weak supervision**: i label derivano dall'origine del dataset, non da annotazione manuale. Questo spiega le performance sotto-target (AUC 0.633 vs target 0.75). Vedi `docs/report/sections/06_pose.tex` per la discussione completa.

### Compatibilità DataParallel
I modelli sono salvati senza prefisso `module.` per compatibilità. Se carichi un modello salvato con DataParallel e ottieni errori di chiavi mancanti:

```python
state_dict = checkpoint['model_state_dict']
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
```

### Workers Multiprocessing
Per evitare errori nei container/server, i notebook usano `NUM_WORKERS=0`. Per training locale puoi aumentare questo valore per velocizzare il data loading.

---

## Metriche Raggiunte

| Modello | Metrica | Target | Ottenuto | Status |
|---------|---------|--------|----------|--------|
| Backbone v2 | mAP@0.5 | > 0.85 | **0.987** | ✓ |
| Collar v2 | mAP@0.5 | > 0.75 | **0.853** | ✓ |
| Skin | Accuracy | > 0.80 | **0.806** | ✓ |
| Pose | AUC-ROC | > 0.75 | 0.633 | ~ |
| Breed | Top-1 Acc | > 0.60 | **0.863** | ✓ |

---

## Contatti

Per problemi con il training o i dataset, aprire una issue su GitHub.
