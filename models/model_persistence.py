"""Model persistence utilities for saving and loading trained ML models."""

import os
from pathlib import Path

import joblib


class ModelRegistry:
    """Manage saving and loading of trained ML models."""

    def __init__(self, model_dir: str = "models"):
        """
        Initialize the model registry.

        Args:
            model_dir: Directory to store models (default: 'models')
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: object, model_name: str, metadata: dict | None = None) -> str:
        """
        Save a trained model to disk.

        Args:
            model: Trained scikit-learn or sklearn-compatible model
            model_name: Name identifier for the model (e.g., 'logistic_v1')
            metadata: Optional dict with training info (accuracy, date, etc.)

        Returns:
            Path to saved model file
        """
        model_path = self.model_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)

        # Save metadata if provided
        if metadata:
            meta_path = self.model_dir / f"{model_name}_meta.txt"
            with open(meta_path, "w") as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")

        return str(model_path)

    def load_model(self, model_name: str) -> object:
        """
        Load a saved model from disk.

        Args:
            model_name: Name identifier of the model (without .pkl extension)

        Returns:
            Loaded model object

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        model_path = self.model_dir / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        return joblib.load(model_path)

    def list_models(self) -> list[str]:
        """
        List all saved models in the registry.

        Returns:
            List of model names (without extensions)
        """
        models = [f.stem for f in self.model_dir.glob("*.pkl")]
        return sorted(models)

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a saved model and its metadata.

        Args:
            model_name: Name identifier of the model

        Returns:
            True if deleted, False if not found
        """
        model_path = self.model_dir / f"{model_name}.pkl"
        meta_path = self.model_dir / f"{model_name}_meta.txt"

        success = False
        if model_path.exists():
            model_path.unlink()
            success = True

        if meta_path.exists():
            meta_path.unlink()

        return success

    def get_model_info(self, model_name: str) -> dict:
        """
        Retrieve metadata about a saved model.

        Args:
            model_name: Name identifier of the model

        Returns:
            Dict with model metadata
        """
        meta_path = self.model_dir / f"{model_name}_meta.txt"

        if not meta_path.exists():
            return {}

        metadata = {}
        with open(meta_path, "r") as f:
            for line in f:
                if ": " in line:
                    key, value = line.strip().split(": ", 1)
                    metadata[key] = value

        return metadata


# Example usage:
# registry = ModelRegistry()
# registry.save_model(trained_model, "logistic_v1", {"accuracy": 0.92, "date": "2024-01-15"})
# loaded_model = registry.load_model("logistic_v1")
# all_models = registry.list_models()
