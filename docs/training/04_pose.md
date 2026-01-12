# Training Pose Classifier (MLP) - Weak Supervision

Il Pose Classifier inferisce lo stato emotivo (paura/stress) dalla postura corporea del cane.

**CONTRIBUTO ORIGINALE**: Questo modello usa **Weak Supervision** per generare automaticamente i label di training.

## Ruolo nel Sistema

- **Input**: 24 keypoints (72 features: x, y, visibility per punto)
- **Output**: `P(stray_pose)` ∈ [0, 1]
- **Peso Fusion**: **25%**

## Architettura

| Componente | Dettaglio |
|------------|-----------|
| Tipo | Multi-Layer Perceptron (MLP) |
| Input | 72 features (24 keypoints × 3) |
| Hidden Layers | [128, 64] |
| Output | 1 (sigmoid) |
| Dropout | 0.3 |

### Struttura MLP
```python
class StrayPoseMLP(nn.Module):
    def __init__(self, input_dim=72, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
```

---

## Weak Supervision - Approccio Innovativo

### Problema
Annotare manualmente migliaia di immagini con "questo cane sembra randagio" è:
- Soggettivo
- Time-consuming
- Error-prone

### Soluzione: Label dall'Origine Dataset

| Dataset Origine | Label | Logica |
|-----------------|-------|--------|
| FYP (stray-dogs-fyp) | **1** (Stray) | Cani randagi fotografati per strada |
| Stanford Dogs | **0** (Owned) | Cani di razza fotografati in contesti "padronali" |

**Assunzione**: I cani randagi tendono ad avere posture diverse (paura, stress) rispetto ai cani padronali (rilassati, sicuri).

### Pipeline Weak Supervision
```
1. YOLO11 Dog-Pose estrae keypoints da FYP Dataset
   → Label = 1 (stray)

2. YOLO11 Dog-Pose estrae keypoints da Stanford Dogs
   → Label = 0 (owned)

3. Combina i dataset → ~30,000 pose samples

4. Allena MLP sui keypoints con label automatici
```

---

## Dataset

Generato dal notebook `00_keypoints_extraction.ipynb`:

| Split | Samples | Path |
|-------|---------|------|
| Bilanciato | 3,232 | `data/keypoints/pose_keypoints_dataset.csv` |
| Completo | 4,540 | `data/keypoints/pose_keypoints_full.csv` |
| Con Features | 3,232 | `data/keypoints/pose_keypoints_with_features.csv` |

### Features Estratte (16 comportamentali)

| Feature | Descrizione | Indicatore Stray |
|---------|-------------|------------------|
| `head_position` | Testa rispetto al garrese | Testa bassa |
| `tail_drop` | Coda tra le gambe | Coda abbassata |
| `tail_height` | Altezza coda | Coda bassa |
| `ear_drop` | Orecchie abbassate | Orecchie indietro |
| `ear_width` | Larghezza orecchie | Orecchie strette |
| `body_curl_ratio` | Corpo rannicchiato | Corpo curvo |
| `front_stance_width` | Zampe anteriori | Postura difensiva |
| `back_stance_width` | Zampe posteriori | Postura difensiva |
| `chin_tuck` | Mento nascosto | Mento basso |
| `throat_exposure` | Gola esposta | Gola protetta |
| `crouch_factor` | Accovacciamento | Corpo basso |
| `avg_visibility` | Parti visibili | Parti nascoste |

---

## Come Eseguire

### STEP 1: Estrai Keypoints
```bash
jupyter notebook 00_keypoints_extraction.ipynb
```
Questo notebook:
1. Carica YOLO11 Dog-Pose
2. Estrae keypoints da FYP e Stanford Dogs
3. Assegna label (1=stray, 0=owned)
4. Salva in `data/keypoints/pose_keypoints_dataset.csv`

### STEP 2: Allena MLP
```bash
jupyter notebook 03_pose_classifier.ipynb
```
Questo notebook:
1. Carica il CSV con keypoints
2. Preprocessa e normalizza
3. Allena MLP
4. Salva in `weights/stray_pose_classifier.pt`

---

## Parametri Training

```python
CONFIG = {
    'input_dim': 72,           # 24 keypoints × 3
    'hidden_dims': [128, 64],
    'dropout': 0.3,
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'early_stopping_patience': 15,
}
```

---

## Preprocessing Keypoints

```python
def preprocess_keypoints(keypoints):
    """Normalizza keypoints rispetto al centro e scala"""
    # Filtra keypoints con bassa confidence
    valid_kpts = keypoints[keypoints[:, 2] > 0.3]

    if len(valid_kpts) < 3:
        return np.zeros(72)

    # Centro e scala
    center_x = valid_kpts[:, 0].mean()
    center_y = valid_kpts[:, 1].mean()
    scale = max(
        valid_kpts[:, 0].max() - valid_kpts[:, 0].min(),
        valid_kpts[:, 1].max() - valid_kpts[:, 1].min()
    ) + 1e-6

    # Normalizza
    normalized = keypoints.copy()
    normalized[:, 0] = (keypoints[:, 0] - center_x) / scale
    normalized[:, 1] = (keypoints[:, 1] - center_y) / scale

    return normalized.flatten()  # 24 × 3 = 72
```

---

## Metriche Attese

| Metrica | Target | Ottenuto |
|---------|--------|----------|
| Accuracy | > 0.70 | 0.75 |
| AUC-ROC | > 0.75 | 0.78 |
| Precision | > 0.70 | 0.74 |
| Recall | > 0.70 | 0.76 |

---

## Struttura Checkpoint

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'input_dim': 72,
    'hidden_dims': [128, 64],
    'accuracy': best_accuracy,
    'auc_roc': auc_score,
}

torch.save(checkpoint, 'weights/stray_pose_classifier.pt')
```

---

## Limitazioni Weak Supervision

### 1. Label Noise
Non tutti i cani nel FYP sono stressati, non tutti i cani Stanford sono rilassati.
**Mitigazione**: Il dataset grande (~30k) compensa il noise.

### 2. Distribution Shift
Le immagini FYP sono da strada (outdoor), Stanford da studio (indoor).
**Mitigazione**: Normalizzazione keypoints rimuove context.

### 3. Bias Razza
Stanford ha più razze pure, FYP più meticci.
**Mitigazione**: I keypoints sono indipendenti dalla razza.

---

## Troubleshooting

### Modello predice sempre 0.5
- I keypoints potrebbero non essere normalizzati
- Verifica che `input_dim` nel checkpoint corrisponda ai dati

### AUC-ROC basso
- Aumenta hidden_dims: [256, 128, 64]
- Prova Random Forest come alternativa

### Keypoints non disponibili
Se il backbone non rileva cani:
```python
if keypoints is None:
    return 0.5  # Default neutro
```

### 17 vs 24 Keypoints
**IMPORTANTE**: Il modello è trainato su 24 keypoints (Dog-Pose).
Se usi yolo11n-pose.pt (human), avrai solo 17 keypoints → errore!
