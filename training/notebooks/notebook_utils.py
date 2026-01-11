"""
Utilità condivise per tutti i notebook di training ResQPet.

Questo modulo fornisce funzioni per:
- Determinare i path relativi indipendentemente dalla posizione di esecuzione
- Auto-detect del device (CUDA/MPS/CPU)
- Struttura directory standard

Uso:
    from notebook_utils import get_paths, get_device
    paths = get_paths()
    device = get_device()
"""

from pathlib import Path
import torch


def get_paths():
    """
    Determina i path del progetto in modo relativo e portabile.

    Funziona indipendentemente da dove viene eseguito il notebook:
    - Dalla cartella notebooks/
    - Dalla cartella training/
    - Dalla cartella ResQPet/
    - Da qualsiasi altra posizione (cerca ResQPet nella struttura)

    Returns:
        dict con i seguenti path:
            - project_dir: ResQPet/
            - base_dir: Directory padre di ResQPet (contiene i dataset)
            - weights_dir: ResQPet/backend/weights/
            - data_dir: ResQPet/data/
            - notebooks_dir: ResQPet/training/notebooks/
            - runs_dir: ResQPet/training/runs/

            Dataset (relativi a base_dir):
            - fyp_dataset: stray-dogs-detection/fypdataset/
            - stanford_dogs: Stanford Dog/
            - skin_dataset: Dog's skin diseases/
            - collar_dataset: Dog with Leash/
    """
    # Determina la directory corrente
    cwd = Path.cwd()

    # Cerca PROJECT_DIR (ResQPet)
    project_dir = None

    # Caso 1: Siamo in notebooks/
    if cwd.name == "notebooks" and (cwd.parent.name == "training"):
        project_dir = cwd.parent.parent
    # Caso 2: Siamo in training/
    elif cwd.name == "training" and (cwd / "notebooks").exists():
        project_dir = cwd.parent
    # Caso 3: Siamo in ResQPet/
    elif cwd.name == "ResQPet" and (cwd / "training" / "notebooks").exists():
        project_dir = cwd
    # Caso 4: Cerca ResQPet nella struttura
    else:
        search_dir = cwd
        while search_dir.parent != search_dir:
            if search_dir.name == "ResQPet" and (search_dir / "training").exists():
                project_dir = search_dir
                break
            # Cerca anche nei figli
            for child in search_dir.iterdir():
                if child.is_dir() and child.name == "ResQPet" and (child / "training").exists():
                    project_dir = child
                    break
            if project_dir:
                break
            search_dir = search_dir.parent

    # Fallback: usa la directory corrente
    if project_dir is None:
        print("⚠️ WARNING: Impossibile trovare la directory ResQPet.")
        print(f"   Directory corrente: {cwd}")
        print("   Usando la directory corrente come base.")
        project_dir = cwd

    # NUOVA STRUTTURA: i dataset sono dentro ResQPet/datasets/
    datasets_dir = project_dir / "datasets"

    # Definisci tutti i path
    paths = {
        # Directory principali
        'project_dir': project_dir,
        'base_dir': project_dir.parent,  # Manteniamo per retrocompatibilità
        'datasets_dir': datasets_dir,

        # Directory di output
        'weights_dir': project_dir / "weights",  # Aggiornato: ora in root, non in backend/
        'data_dir': project_dir / "data",
        'notebooks_dir': project_dir / "training" / "notebooks",
        'runs_dir': project_dir / "training" / "runs",

        # Dataset - NUOVI PATH (tutti in datasets/)
        'dog_pose_dataset': datasets_dir / "dog-pose",
        'fyp_dataset': datasets_dir / "stray-dogs-fyp",
        'stanford_dogs': datasets_dir / "stanford-dogs",
        'skin_dataset': datasets_dir / "dog-skin-diseases",
        'collar_dataset': datasets_dir / "dog-with-leash",

        # Alias per retrocompatibilità
        'keypoints_dir': project_dir / "data" / "keypoints",
    }

    # Crea directory di output se non esistono
    for key in ['weights_dir', 'data_dir', 'runs_dir', 'keypoints_dir']:
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths


def get_device():
    """
    Auto-detect del miglior device disponibile.

    Returns:
        torch.device: 'cuda' se disponibile, altrimenti 'mps' (Apple Silicon), altrimenti 'cpu'

    Also returns the device name as string for YOLO and other frameworks.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = 'cuda'
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'mps'
    else:
        device = torch.device('cpu')
        device_name = 'cpu'

    return device, device_name


def print_paths(paths):
    """Stampa tutti i path in modo formattato."""
    print("=" * 60)
    print("CONFIGURAZIONE PATH")
    print("=" * 60)

    print("\n📁 Directory Progetto:")
    print(f"   Project:   {paths['project_dir']}")
    print(f"   Base:      {paths['base_dir']}")

    print("\n📤 Directory Output:")
    print(f"   Weights:   {paths['weights_dir']}")
    print(f"   Data:      {paths['data_dir']}")
    print(f"   Runs:      {paths['runs_dir']}")

    print("\n📊 Dataset:")
    dataset_keys = ['dog_pose_dataset', 'fyp_dataset', 'stanford_dogs', 'skin_dataset', 'collar_dataset']
    for key in dataset_keys:
        if key in paths:
            exists = "✓" if paths[key].exists() else "✗"
            print(f"   {exists} {key}: {paths[key]}")

    print("=" * 60)


def check_dataset_exists(paths, dataset_key):
    """
    Verifica se un dataset esiste e stampa info.

    Args:
        paths: dict ritornato da get_paths()
        dataset_key: chiave del dataset (es. 'fyp_dataset', 'stanford_dogs')

    Returns:
        bool: True se il dataset esiste
    """
    path = paths.get(dataset_key)
    if path is None:
        print(f"⚠️ Chiave dataset sconosciuta: {dataset_key}")
        return False

    if path.exists():
        # Conta immagini
        images = list(path.rglob("*.jpg")) + list(path.rglob("*.png")) + list(path.rglob("*.jpeg"))
        print(f"✓ {dataset_key}: {len(images)} immagini trovate")
        return True
    else:
        print(f"✗ {dataset_key}: Directory non trovata")
        print(f"  Path: {path}")
        return False


# Test se eseguito direttamente
if __name__ == "__main__":
    print("Test notebook_utils.py\n")

    paths = get_paths()
    print_paths(paths)

    device, device_name = get_device()
    print(f"\nDevice: {device} ({device_name})")

    print("\nVerifica dataset:")
    for key in ['dog_pose_dataset', 'fyp_dataset', 'stanford_dogs', 'skin_dataset', 'collar_dataset']:
        check_dataset_exists(paths, key)
