"""
Breed Classifier
Classifies dog breed to estimate stray probability based on breed statistics
"""

import numpy as np
import cv2
import json
from typing import Dict, Optional, Tuple, List
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class BreedClassifierModel(nn.Module):
    """
    EfficientNet-B0 based breed classifier.
    Architecture matches notebook 04_breed_classifier.ipynb
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()

        # Backbone EfficientNet-B0 (without classifier head)
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=False,
            num_classes=0  # Remove built-in classifier
        )

        # Feature dimension
        self.feature_dim = self.backbone.num_features

        # Custom classifier (matching notebook architecture)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class BreedClassifier:
    """
    EfficientNet-based breed classifier using Stanford Dogs dataset.

    Output: (breed, confidence) -> Used with breed priors to get P(stray|breed)
    """

    # Grouped breed categories (simplified from 120 to 20)
    BREED_GROUPS = {
        'pitbull_amstaff': ['American_Staffordshire_terrier', 'Staffordshire_bullterrier',
                           'American_pit_bull_terrier'],
        'shepherd': ['German_shepherd', 'Belgian_malinois', 'Australian_shepherd',
                     'Border_collie', 'Shetland_sheepdog'],
        'retriever': ['Labrador_retriever', 'Golden_retriever', 'Flat-coated_retriever',
                      'Chesapeake_Bay_retriever'],
        'hound': ['Beagle', 'Basset', 'Bloodhound', 'Dachshund', 'Greyhound'],
        'terrier': ['Yorkshire_terrier', 'West_Highland_white_terrier', 'Scottish_terrier',
                    'Bull_terrier', 'Fox_terrier'],
        'toy': ['Chihuahua', 'Maltese', 'Pomeranian', 'Toy_poodle', 'Papillon'],
        'working': ['Rottweiler', 'Doberman', 'Boxer', 'Great_Dane', 'Mastiff'],
        'spitz': ['Husky', 'Malamute', 'Samoyed', 'Akita', 'Chow_chow'],
        'bulldog': ['Bulldog', 'French_bulldog', 'Boston_terrier'],
        'poodle': ['Standard_poodle', 'Miniature_poodle'],
        'mixed': ['Mixed_breed', 'Unknown']
    }

    # Breed priors: P(stray|breed_group) based on shelter statistics
    BREED_PRIORS = {
        'pitbull_amstaff': 0.75,  # High shelter presence
        'shepherd': 0.50,
        'retriever': 0.25,  # Low - popular family dogs
        'hound': 0.55,
        'terrier': 0.40,
        'toy': 0.20,  # Low - expensive, rarely abandoned
        'working': 0.50,
        'spitz': 0.35,
        'bulldog': 0.30,
        'poodle': 0.25,
        'mixed': 0.70,  # High - most common in shelters
        'unknown': 0.50
    }

    def __init__(self, model_path: Optional[str] = None,
                 priors_path: Optional[str] = None,
                 device: str = 'auto'):
        """
        Initialize breed classifier.

        Args:
            model_path: Path to trained EfficientNet weights
            priors_path: Path to breed priors JSON file
            device: 'cuda', 'cpu', or 'auto'
        """
        self.model = None
        self.device = self._get_device(device)
        self.class_names = []

        # Load breed priors
        if priors_path and Path(priors_path).exists():
            with open(priors_path, 'r') as f:
                self.BREED_PRIORS = json.load(f)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]) if TORCH_AVAILABLE else None

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _get_device(self, device: str) -> str:
        if not TORCH_AVAILABLE:
            return 'cpu'
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _load_model(self, model_path: str):
        """Load trained model weights"""
        if not TORCH_AVAILABLE or not TIMM_AVAILABLE:
            print("[BreedClassifier] PyTorch/timm not available")
            return

        try:
            # Load checkpoint (with weights_only=False for PyTorch 2.6+)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Try different keys for class names (and determine num_classes from them)
            if 'class_names' in checkpoint:
                self.class_names = checkpoint['class_names']
            elif 'categories' in checkpoint:
                self.class_names = checkpoint['categories']
            else:
                self.class_names = list(self.BREED_GROUPS.keys())

            # Get num_classes from checkpoint or infer from class_names
            num_classes = checkpoint.get('num_classes', len(self.class_names))

            # Load breed priors from checkpoint if available
            if 'breed_priors' in checkpoint:
                self.BREED_PRIORS = checkpoint['breed_priors']

            # Create model with custom architecture (matching notebook 04)
            self.model = BreedClassifierModel(num_classes=num_classes)

            # Extract model_state_dict from checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            print(f"[BreedClassifier] Model loaded from {model_path}")

        except Exception as e:
            print(f"[BreedClassifier] Failed to load model: {e}")
            self.model = None

    def predict(self, roi: np.ndarray) -> Tuple[str, float]:
        """
        Predict dog breed.

        Args:
            roi: Cropped image of the dog (BGR)

        Returns:
            (breed_group, confidence)
        """
        if roi is None or roi.size == 0:
            return ('unknown', 0.5)

        if self.model is None:
            return self._heuristic_prediction(roi)

        try:
            # Preprocess
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]

            # Get top prediction
            top_idx = torch.argmax(probs).item()
            confidence = float(probs[top_idx])
            breed = self.class_names[top_idx] if top_idx < len(self.class_names) else 'unknown'

            return (breed, confidence)

        except Exception as e:
            print(f"[BreedClassifier] Prediction error: {e}")
            return self._heuristic_prediction(roi)

    def _heuristic_prediction(self, roi: np.ndarray) -> Tuple[str, float]:
        """
        Simple heuristic breed estimation based on size/color.
        Very basic - mainly for testing without model.
        """
        if roi is None or roi.size == 0:
            return ('unknown', 0.3)

        try:
            h, w = roi.shape[:2]

            # Color analysis
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_value = np.mean(hsv[:, :, 2])  # Brightness
            avg_saturation = np.mean(hsv[:, :, 1])

            # Size-based rough estimate
            area = h * w

            if area < 10000:  # Small dog
                return ('toy', 0.3)
            elif area > 50000:  # Large dog
                if avg_value < 100:  # Dark colored
                    return ('working', 0.3)
                else:
                    return ('retriever', 0.3)
            else:  # Medium
                return ('mixed', 0.4)

        except Exception:
            return ('unknown', 0.3)

    def get_stray_probability(self, roi: np.ndarray) -> float:
        """
        Get P(stray|breed) for the detected dog.

        Args:
            roi: Dog image

        Returns:
            P(stray|breed) based on breed classification and priors
        """
        breed, confidence = self.predict(roi)

        # Get prior for breed group
        p_stray_breed = self.BREED_PRIORS.get(breed, 0.5)

        # Weight by confidence
        # Low confidence -> move toward 0.5 (uncertain)
        weighted_prob = p_stray_breed * confidence + 0.5 * (1 - confidence)

        return weighted_prob

    def predict_with_details(self, roi: np.ndarray) -> Dict:
        """
        Predict with full details.

        Args:
            roi: Dog image

        Returns:
            Dictionary with breed prediction and stray probability
        """
        breed, confidence = self.predict(roi)
        p_stray = self.get_stray_probability(roi)

        return {
            'breed': breed,
            'confidence': confidence,
            'p_stray_breed': p_stray,
            'breed_prior': self.BREED_PRIORS.get(breed, 0.5),
            'top_5': self._get_top_k(roi, k=5) if self.model else []
        }

    def _get_top_k(self, roi: np.ndarray, k: int = 5) -> List[Dict]:
        """Get top-k breed predictions"""
        if self.model is None or roi is None:
            return []

        try:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]

            top_k_values, top_k_indices = torch.topk(probs, min(k, len(self.class_names)))

            results = []
            for val, idx in zip(top_k_values, top_k_indices):
                breed = self.class_names[idx] if idx < len(self.class_names) else 'unknown'
                results.append({
                    'breed': breed,
                    'probability': float(val),
                    'prior': self.BREED_PRIORS.get(breed, 0.5)
                })

            return results

        except Exception:
            return []
