"""
Stray Pose Classifier
Classifies dog posture as stray-like or owned-like using weak supervision approach.

Training Strategy (Weak Supervision):
- Stray (label=1): Keypoints from FYP Dataset (known stray dogs)
- Owned (label=0): Keypoints from Stanford Dogs (owned/show dogs)

No manual annotation required - labels derived from dataset source.
"""

import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class StrayPoseMLP(nn.Module):
    """
    MLP classifier for stray pose detection.

    Input: Flattened normalized keypoints
           - Dog-Pose model: 24 points × 3 values = 72 features (RECOMMENDED)
           - COCO fallback:  17 points × 3 values = 51 features (human keypoints, NOT recommended)

    Output: P(stray_pose) - probability of stray-like posture

    Architecture: Linear → ReLU → BatchNorm → Dropout (matching notebook 03)

    NOTE: The input_dim is read from the model checkpoint, so it will match
    whatever was used during training. Default is 72 for dog-pose model.
    """

    def __init__(self, input_dim: int = 72, hidden_dims: List[int] = [128, 64], dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class StrayPoseClassifier:
    """
    Classifier for detecting stray-like postures from keypoints.

    Behavioral indicators of stray dogs:
    - Tail tucked between legs (fear)
    - Body curled/defensive posture
    - Head lowered
    - Ears back/flattened
    - Spine curved (submissive)
    - Wide/defensive leg stance

    Training approach: Weak supervision using dataset source as label.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize pose classifier.

        Args:
            model_path: Path to trained MLP weights
            device: 'cuda', 'cpu', or 'auto'
        """
        self.model = None
        self.device = self._get_device(device)

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
        if not TORCH_AVAILABLE:
            print("[StrayPoseClassifier] PyTorch not available")
            return

        try:
            # Load checkpoint (with weights_only=False for PyTorch 2.6+)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Determine input dimension and hidden dims from checkpoint
            input_dim = checkpoint.get('input_dim', 51)
            hidden_dims = checkpoint.get('hidden_dims', [128, 64])

            self.model = StrayPoseMLP(input_dim=input_dim, hidden_dims=hidden_dims)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            print(f"[StrayPoseClassifier] Model loaded from {model_path}")

        except Exception as e:
            print(f"[StrayPoseClassifier] Failed to load model: {e}")
            self.model = None

    def predict(self, keypoints: np.ndarray) -> float:
        """
        Predict probability of stray-like posture.

        Args:
            keypoints: Normalized keypoints array (N, 3) where each row is (x, y, visibility)

        Returns:
            P(stray_pose) in [0, 1]
            Higher value = more stray-like posture
        """
        if keypoints is None or len(keypoints) == 0:
            return 0.5  # Uncertain

        # Extract handcrafted features
        features = self._extract_pose_features(keypoints)

        if self.model is None:
            return self._heuristic_classification(features)

        try:
            # Flatten keypoints for MLP input
            flat_kpts = keypoints.flatten()

            # Convert to tensor
            tensor = torch.FloatTensor(flat_kpts).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                p_stray = float(self.model(tensor)[0])

            return p_stray

        except Exception as e:
            print(f"[StrayPoseClassifier] Prediction error: {e}")
            return self._heuristic_classification(features)

    def _extract_pose_features(self, keypoints: np.ndarray) -> Dict:
        """
        Extract interpretable pose features from keypoints.

        These features capture behavioral indicators of stray dogs.
        """
        features = {}

        if keypoints is None or len(keypoints) < 5:
            return features

        try:
            # Get key body parts (indices depend on keypoint format)
            # Using COCO-style indices as default
            kpts = keypoints

            # Feature 1: Head position relative to body center
            # Low head = submissive/fearful
            if len(kpts) > 0:
                nose = kpts[0] if kpts[0, 2] > 0.3 else None
                # Approximate body center
                body_y = np.mean([k[1] for k in kpts if k[2] > 0.3])
                if nose is not None:
                    features['head_low'] = float(nose[1] > body_y)

            # Feature 2: Body width ratio (defensive = wider stance)
            left_points = [k for i, k in enumerate(kpts) if i % 2 == 1 and k[2] > 0.3]
            right_points = [k for i, k in enumerate(kpts) if i % 2 == 0 and k[2] > 0.3]

            if left_points and right_points:
                left_x = np.mean([p[0] for p in left_points])
                right_x = np.mean([p[0] for p in right_points])
                features['stance_width'] = abs(left_x - right_x)

            # Feature 3: Keypoint visibility (stressed dogs may have more hidden parts)
            avg_visibility = np.mean(kpts[:, 2])
            features['avg_visibility'] = avg_visibility

            # Feature 4: Vertical compactness (curled up = smaller y range)
            visible_kpts = kpts[kpts[:, 2] > 0.3]
            if len(visible_kpts) > 2:
                y_range = np.max(visible_kpts[:, 1]) - np.min(visible_kpts[:, 1])
                x_range = np.max(visible_kpts[:, 0]) - np.min(visible_kpts[:, 0])
                if x_range > 0:
                    features['body_curl'] = y_range / x_range

            # Feature 5: Symmetry (stressed postures are often asymmetric)
            if len(left_points) >= 2 and len(right_points) >= 2:
                left_y = np.std([p[1] for p in left_points])
                right_y = np.std([p[1] for p in right_points])
                features['asymmetry'] = abs(left_y - right_y)

        except Exception as e:
            print(f"[StrayPoseClassifier] Feature extraction error: {e}")

        return features

    def _heuristic_classification(self, features: Dict) -> float:
        """
        Rule-based classification when model not available.

        Args:
            features: Extracted pose features

        Returns:
            P(stray_pose) estimate
        """
        if not features:
            return 0.5

        score = 0.3  # Base probability

        # Head position
        if features.get('head_low', False):
            score += 0.15

        # Body curl (smaller = more curled)
        body_curl = features.get('body_curl', 1.0)
        if body_curl < 0.8:
            score += 0.15

        # Wide stance (defensive)
        stance_width = features.get('stance_width', 0.5)
        if stance_width > 0.6:
            score += 0.1

        # Asymmetry (stressed)
        asymmetry = features.get('asymmetry', 0.0)
        if asymmetry > 0.1:
            score += 0.1

        # Low visibility (hiding parts)
        avg_vis = features.get('avg_visibility', 0.8)
        if avg_vis < 0.6:
            score += 0.1

        return min(0.9, score)

    def predict_with_details(self, keypoints: np.ndarray) -> Dict:
        """
        Predict with detailed feature breakdown.

        Args:
            keypoints: Normalized keypoints array

        Returns:
            Dictionary with prediction and feature analysis
        """
        p_stray = self.predict(keypoints)
        features = self._extract_pose_features(keypoints)

        return {
            'p_stray_pose': p_stray,
            'p_owned_pose': 1.0 - p_stray,
            'features': features,
            'classification': 'stray_like' if p_stray > 0.5 else 'owned_like',
            'confidence': abs(p_stray - 0.5) * 2
        }


# Training utilities for weak supervision approach
def create_training_dataset(
    stray_keypoints: List[np.ndarray],
    owned_keypoints: List[np.ndarray]
) -> tuple:
    """
    Create balanced training dataset from weak supervision sources.

    Args:
        stray_keypoints: Keypoints from FYP Dataset (stray dogs)
        owned_keypoints: Keypoints from Stanford Dogs (owned dogs)

    Returns:
        X: Feature array
        y: Labels (1=stray, 0=owned)
    """
    # Balance classes
    min_samples = min(len(stray_keypoints), len(owned_keypoints))

    stray_sample = np.random.choice(len(stray_keypoints), min_samples, replace=False)
    owned_sample = np.random.choice(len(owned_keypoints), min_samples, replace=False)

    X = []
    y = []

    for idx in stray_sample:
        X.append(stray_keypoints[idx].flatten())
        y.append(1)

    for idx in owned_sample:
        X.append(owned_keypoints[idx].flatten())
        y.append(0)

    return np.array(X), np.array(y)
