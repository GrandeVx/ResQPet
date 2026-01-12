# Training Collar Detector (YOLOv8n)

Il Collar Detector rileva la presenza/assenza di collare, pettorina o guinzaglio.

## Ruolo nel Sistema

- **Input**: ROI (crop) del cane dalla bbox del backbone
- **Output**: `P(no_collar)` ∈ [0, 1]
- **Peso Fusion**: **35%** (indicatore più forte)

## Architettura

| Componente | Dettaglio |
|------------|-----------|
| Base Model | YOLOv8n (nano) |
| Task | Object Detection |
| Classi | 2 (Dog-with-Leash, Dog-without-Leash) |
| Input Size | 640×640 |

---

## Versioni

### v2 (Attuale) - 2026-01-03

Trainato sul dataset merged dalla piattaforma di labeling ResQPet.

| Metrica | Valore |
|---------|--------|
| mAP@0.5 | **0.853** |
| mAP@0.5:0.95 | 0.722 |
| Precision | 0.741 |
| Recall | 0.854 |

**Dataset**: 7,576 immagini annotate da 2 utenti sulla piattaforma di labeling.

| Split | Immagini |
|-------|----------|
| Train | 6,061 |
| Val | 1,516 |

**Training**:
- 100 epochs, batch 128
- 2x NVIDIA RTX 5090 (multi-GPU)
- Mixed precision FP16
- ~20 minuti totali

### v1 (Legacy)

Trainato su dataset Roboflow "dog-with-leash" (152 immagini).

| Metrica | Valore |
|---------|--------|
| mAP@0.5 | 0.51 |
| Precision | 0.48 |
| Recall | 0.54 |

---

## Dataset

### Piattaforma di Labeling (Raccomandato)

Per creare/aggiornare il dataset dal labeling:

```bash
# 1. Esporta annotazioni JSON dalla piattaforma
# 2. Rinomina i file: user_<id>_annotations_<version>.json

# 3. Merge in formato YOLO
python -m labeling_tool.scripts.merge_exports \
    --json-dir labeling_data/exports/json \
    --output labeling_data/exports/collar_yolo \
    --train-split 0.8
```

**Classi**:
- `Dog-with-Leash` (classe 0) - cane con collare/guinzaglio visibile
- `Dog-without-Leash` (classe 1) - cane senza collare

### Roboflow (Fallback)

Se non hai dati dalla piattaforma, il notebook usa automaticamente il dataset Roboflow:

| Split | Immagini | Path |
|-------|----------|------|
| Train | 106 | `datasets/dog-with-leash/split_dataset/train/` |
| Val | 30 | `datasets/dog-with-leash/split_dataset/valid/` |
| Test | 16 | `datasets/dog-with-leash/split_dataset/test/` |

---

## Parametri Training

Il notebook rileva automaticamente l'hardware e configura i parametri:

### Multi-GPU (2x RTX 5090)
```python
EPOCHS = 100
BATCH_SIZE = 128  # 64 per GPU
DEVICE = [0, 1]
WORKERS = 8
```

### Single GPU
```python
EPOCHS = 100
BATCH_SIZE = 64
DEVICE = 0
WORKERS = 4
```

### Apple Silicon (MPS)
```python
EPOCHS = 100
BATCH_SIZE = 16
DEVICE = 'mps'
WORKERS = 4
```

### Augmentation
```python
training_args = {
    'hsv_h': 0.02,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 15,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 5,
    'flipud': 0.2,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.1,
    'amp': True,  # Mixed precision
}
```

---

## Come Eseguire

### 1. Prepara il dataset (opzionale)

Se hai nuove annotazioni dalla piattaforma:
```bash
python -m labeling_tool.scripts.merge_exports \
    --json-dir labeling_data/exports/json \
    --output labeling_data/exports/collar_yolo
```

### 2. Apri il notebook
```bash
jupyter lab training/notebooks/01_collar_detector.ipynb
```

### 3. Esegui tutte le celle

Il notebook:
- Rileva automaticamente il dataset (merged o Roboflow)
- Configura hardware (multi-GPU, single GPU, MPS, CPU)
- Esegue training
- Salva il modello in `weights/collar_detector.pt`

### 4. Tempo stimato

| Hardware | Tempo |
|----------|-------|
| 2x RTX 5090 | ~20 min |
| RTX 4090 | ~40 min |
| RTX 3080 | ~1h |
| Apple M1/M2 | ~2-3h |

---

## Logica Output

```python
def get_no_collar_probability(results):
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 0:  # Dog-with-Leash
            return 1.0 - conf  # Bassa prob. di essere senza
        elif cls == 1:  # Dog-without-Leash
            return conf  # Alta prob. di essere senza

    return 0.7  # Default se nessuna detection
```

---

## Metriche

### v2 (Attuale)

| Metrica | Target | Ottenuto |
|---------|--------|----------|
| mAP@0.5 | > 0.80 | **0.853** |
| mAP@0.5:0.95 | > 0.60 | 0.722 |
| Precision | > 0.70 | 0.741 |
| Recall | > 0.80 | **0.854** |

Il recall alto (85.4%) significa che il modello trova la maggior parte dei cani.
La precision (74.1%) indica alcuni false positive, migliorabile con più dati.

---

## Analisi Risultati

Dopo il training, verifica:
- `training/runs/collar_detector/confusion_matrix.png`
- `training/runs/collar_detector/results.csv`
- `training/runs/collar_detector/training_curves.png`

---

## Miglioramenti Futuri

### 1. Più dati dalla piattaforma
Continua ad annotare immagini sulla piattaforma di labeling.
Ogni 1,000 nuove annotazioni, riaddestra il modello.

### 2. Active Learning
Usa il modello per pre-annotare nuove immagini, poi correggi manualmente.

### 3. Hard Negative Mining
Identifica i casi dove il modello sbaglia e aggiungi al training set.

---

## Troubleshooting

### mAP non migliora
- Verifica che le annotazioni siano consistenti
- Controlla il bilanciamento delle classi
- Prova learning rate più basso (0.0005)

### Out of Memory
- Riduci batch size
- Riduci workers
- Disabilita cache RAM

### Training lento
- Abilita `amp=True` (mixed precision)
- Usa `cache='ram'` se hai RAM sufficiente
- Aumenta workers

---

## Changelog

| Data | Versione | Note |
|------|----------|------|
| 2026-01-03 | v2 | Retrained su 7,576 immagini dalla piattaforma labeling. mAP 0.51→0.85 |
| 2025-12-XX | v1 | Training iniziale su Roboflow dataset (152 img) |
