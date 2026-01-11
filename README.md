# ResQPet 🐕

Sistema di rilevamento cani randagi basato su Computer Vision e Deep Learning.

## Descrizione

ResQPet è un sistema di monitoraggio intelligente che utilizza telecamere CCTV per identificare e classificare cani randagi in tempo reale. Il sistema combina diversi modelli di machine learning per analizzare:

- **Rilevamento cani** - Backbone YOLO per detection
- **Presenza collare** - Classificatore collar/no-collar
- **Condizione pelo** - Analisi stato di salute
- **Postura** - Classificazione comportamento (normale/spaventato)
- **Razza** - Identificazione razza per segnalazioni

## Struttura Progetto

```
ResQPet/
├── training/
│   └── notebooks/          # Notebook di training
├── weights/                # Modelli addestrati
├── backend/                # API FastAPI
├── frontend/               # UI React
├── labeling_tool/          # Tool annotazione dati
└── docs/                   # Documentazione
```

## Setup

```bash
# Crea virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure: venv\Scripts\activate  # Windows

# Installa dipendenze
pip install -r requirements.txt
```

## Training Notebooks

I notebook nella cartella `training/notebooks/` seguono un ordine sequenziale:

1. `00_keypoints_extraction.ipynb` - Estrazione keypoints pose
2. `00a_yolo_dog_pose_training.ipynb` - Training YOLO pose
3. `01_collar_detector.ipynb` - Training collar detector
4. `02_skin_classifier.ipynb` - Training skin classifier
5. `03_pose_classifier.ipynb` - Training pose classifier
6. `04_breed_classifier.ipynb` - Training breed classifier
7. `05_pipeline_demo.ipynb` - Demo pipeline completa

## Team

Progetto universitario sviluppato per il corso di Machine Learning.

## License

MIT
