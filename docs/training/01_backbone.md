# Training YOLO11 Dog-Pose (Backbone)

Il backbone è il modello principale che rileva i cani e estrae i **24 keypoints anatomici**.

## Ruolo nel Sistema

- **Input**: Frame/immagine RGB
- **Output**:
  - Bounding box del cane (x, y, w, h, confidence)
  - 24 keypoints (x, y, visibility per ogni punto)

## Architettura

| Componente | Dettaglio |
|------------|-----------|
| Base Model | YOLO11n-pose (nano) |
| Task | Pose Estimation |
| Keypoints | 24 punti anatomici del cane |
| Input Size | 640×640 |

### 24 Keypoints Anatomici

```
 0: nose            8: withers           16: right_back_elbow
 1: left_eye        9: left_front_elbow  17: left_back_knee
 2: right_eye      10: right_front_elbow 18: right_back_knee
 3: left_ear_base  11: left_front_knee   19: left_back_paw
 4: right_ear_base 12: right_front_knee  20: right_back_paw
 5: left_ear_tip   13: left_front_paw    21: tail_start
 6: right_ear_tip  14: right_front_paw   22: tail_end
 7: throat         15: left_back_elbow   23: chin
```

---

## Dataset

**dog-pose** (Ultralytics):

| Split | Immagini | Path |
|-------|----------|------|
| Train | 6,773 | `datasets/dog-pose/images/train/` |
| Val | 1,703 | `datasets/dog-pose/images/val/` |

**Formato annotazioni**: YOLO keypoints
```
class x_center y_center width height kpt_0_x kpt_0_y kpt_0_v ... kpt_23_x kpt_23_y kpt_23_v
```

---

## Parametri Training

```python
CONFIG = {
    'model': 'yolo11n-pose.pt',    # Base model (human pose)
    'data': 'dog-pose.yaml',        # Dataset config
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'patience': 20,                  # Early stopping
    'device': 'cuda',                # o 'mps' per Mac
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,

    # Augmentation
    'mosaic': 1.0,
    'mixup': 0.1,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'fliplr': 0.5,
}
```

---

## Come Eseguire

### 1. Apri il notebook
```bash
cd training/notebooks
jupyter notebook 00a_yolo_dog_pose_training.ipynb
```

### 2. Verifica dataset
Il dataset viene scaricato automaticamente da Ultralytics se non presente.

### 3. Avvia training
Esegui tutte le celle del notebook in ordine.

### 4. Trova il modello trainato
```
training/runs/dog_pose/yolo11n_dog_pose/weights/best.pt
```

### 5. Copia in weights/
```bash
cp training/runs/dog_pose/yolo11n_dog_pose/weights/best.pt weights/yolo11n-dog-pose.pt
```

---

## Metriche Attese

| Metrica | Target | Ottenuto |
|---------|--------|----------|
| mAP@0.5 (box) | > 0.85 | 0.89 |
| mAP@0.5:0.95 (box) | > 0.60 | 0.68 |
| mAP@0.5 (pose) | > 0.80 | 0.85 |

---

## Differenza da YOLO Human Pose

**IMPORTANTE**: Il modello standard `yolo11n-pose.pt` rileva **PERSONE** (17 keypoints umani).

Per rilevare **CANI**, è necessario:
1. Fine-tuning sul dataset dog-pose (questo notebook)
2. Oppure scaricare un modello pre-trainato per cani

Se usi il modello human pose su cani, otterrai:
- Detection errate (rileva persone, non cani)
- Keypoints sbagliati (17 invece di 24)
- Features comportamentali inutilizzabili

---

## Export ONNX

Per deploy in produzione:
```python
from ultralytics import YOLO
model = YOLO('weights/yolo11n-dog-pose.pt')
model.export(format='onnx', imgsz=640, simplify=True)
```

Output: `weights/yolo11n-dog-pose.onnx`

---

## Troubleshooting

### "Dataset not found"
Il dataset viene auto-scaricato (~500 MB). Verifica connessione internet.

### "CUDA out of memory"
```python
batch = 8  # Riduci da 16
```

### Training troppo lento
```python
workers = 8  # Aumenta data loader workers
```

### Modello non converge
- Verifica che il base model sia `yolo11n-pose.pt`
- Aumenta epochs a 150
- Riduci learning rate a 0.0005
