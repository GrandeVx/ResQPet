"""
Skin Disease Classifier
Identifies skin pathologies indicative of neglect/stray status
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
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


class SkinClassifier:
    """
    ResNet50-based classifier for dog skin diseases.

    Classes:
        - Healthy: Normal skin
        - Fungal_infections: Fungal infection (ringworm)
        - Dermatitis: General dermatitis
        - Hypersensitivity: Allergic reactions
        - Demodicosis: Demodex mite infection
        - Ringworm: Tinea (fungal)

    Output: P(disease) - probability of skin disease
    """

    CLASSES = [
        'Dermatitis',
        'Fungal_infections',
        'Healthy',
        'Hypersensitivity',
        'demodicosis',
        'ringworm'
    ]

    # Disease severity weights (how indicative of neglect)
    DISEASE_WEIGHTS = {
        'Healthy': 0.0,
        'Dermatitis': 0.6,
        'Fungal_infections': 0.7,
        'Hypersensitivity': 0.4,
        'demodicosis': 0.8,
        'ringworm': 0.75
    }

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize skin classifier.

        Args:
            model_path: Path to trained model weights
            device: 'cuda', 'cpu', or 'auto'
        """
        self.model = None
        self.device = self._get_device(device)

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
            print("[SkinClassifier] PyTorch/timm not available")
            return

        try:
            # Load checkpoint (with weights_only=False for PyTorch 2.6+)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Get number of classes from checkpoint if available
            num_classes = checkpoint.get('num_classes', len(self.CLASSES))
            if 'class_names' in checkpoint:
                self.CLASSES = checkpoint['class_names']

            # Create ResNet50 model
            self.model = timm.create_model(
                'resnet50',
                pretrained=False,
                num_classes=num_classes
            )

            # Extract model_state_dict from checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            print(f"[SkinClassifier] Model loaded from {model_path}")

        except Exception as e:
            print(f"[SkinClassifier] Failed to load model: {e}")
            self.model = None

    def predict(self, roi: np.ndarray) -> float:
        """
        Predict probability of skin disease.

        Args:
            roi: Cropped image of the dog (BGR)

        Returns:
            P(disease) in [0, 1]
            Higher value = more likely has skin disease
        """
        if roi is None or roi.size == 0:
            return 0.3  # Default: slight disease probability

        if self.model is None:
            return self._heuristic_detection(roi)

        try:
            # Preprocess
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]

            # Calculate weighted disease probability
            p_disease = 0.0
            for i, class_name in enumerate(self.CLASSES):
                weight = self.DISEASE_WEIGHTS.get(class_name, 0.5)
                p_disease += float(probs[i]) * weight

            return min(1.0, p_disease)

        except Exception as e:
            print(f"[SkinClassifier] Prediction error: {e}")
            return self._heuristic_detection(roi)

    def _heuristic_detection(self, roi: np.ndarray) -> float:
        """
        Heuristic skin analysis based on color/texture.

        Args:
            roi: Dog image

        Returns:
            P(disease) estimate
        """
        if roi is None or roi.size == 0:
            return 0.3

        try:
            # Resize for consistent analysis
            img = cv2.resize(roi, (224, 224))

            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Look for red/pink areas (inflammation)
            # Red hue range in HSV
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2

            red_ratio = np.sum(red_mask > 0) / red_mask.size

            # Check for patchy/irregular texture (fungal indicators)
            # Using edge detection as proxy for texture irregularity
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Combine heuristics
            disease_score = 0.2  # Base probability

            if red_ratio > 0.05:
                disease_score += 0.3  # Redness detected
            if edge_density > 0.15:
                disease_score += 0.2  # High texture irregularity

            return min(0.9, disease_score)

        except Exception as e:
            print(f"[SkinClassifier] Heuristic error: {e}")
            return 0.3

    def predict_with_details(self, roi: np.ndarray) -> Dict:
        """
        Predict with detailed class probabilities.

        Args:
            roi: Dog image

        Returns:
            Dictionary with detailed predictions
        """
        p_disease = self.predict(roi)

        # If model available, get class breakdown
        class_probs = {}
        if self.model is not None and roi is not None and roi.size > 0:
            try:
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                tensor = self.transform(rgb).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = torch.softmax(outputs, dim=1)[0]

                for i, class_name in enumerate(self.CLASSES):
                    class_probs[class_name] = float(probs[i])

            except Exception:
                pass

        return {
            'p_disease': p_disease,
            'p_healthy': 1.0 - p_disease,
            'class_probabilities': class_probs,
            'predicted_class': max(class_probs, key=class_probs.get) if class_probs else 'Unknown'
        }
