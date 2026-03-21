"""Unit tests for ML services."""

import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.services.evaluation_service import (
    calculate_classification_metrics,
    calculate_train_test_accuracy,
)
from app.services.model_service import (
    train_decision_tree_classifier,
    train_logistic_classifier,
    train_random_forest_classifier,
)


class TestModelService(unittest.TestCase):
    """Test model training functions."""

    @classmethod
    def setUpClass(cls):
        """Create sample classification dataset for testing."""
        np.random.seed(42)
        cls.X = pd.DataFrame(
            np.random.randn(100, 4),
            columns=["feature_1", "feature_2", "feature_3", "feature_4"],
        )
        cls.y = pd.Series(np.random.randint(0, 2, 100), name="target")
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X,
            cls.y,
            test_size=0.2,
            random_state=42,
        )

    def test_logistic_classifier_trains(self):
        """Test that logistic regression trains without errors."""
        y_train_pred, y_test_pred = train_logistic_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
        )

        self.assertEqual(len(y_train_pred), len(self.y_train))
        self.assertEqual(len(y_test_pred), len(self.y_test))
        self.assertTrue(all(pred in [0, 1] for pred in y_test_pred))

    def test_decision_tree_classifier_trains(self):
        """Test that decision tree trains without errors."""
        y_train_pred, y_test_pred = train_decision_tree_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            max_depth=5,
        )

        self.assertEqual(len(y_train_pred), len(self.y_train))
        self.assertEqual(len(y_test_pred), len(self.y_test))

    def test_random_forest_classifier_trains(self):
        """Test that random forest trains without errors."""
        y_train_pred, y_test_pred = train_random_forest_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
        )

        self.assertEqual(len(y_train_pred), len(self.y_train))
        self.assertEqual(len(y_test_pred), len(self.y_test))


class TestEvaluationService(unittest.TestCase):
    """Test evaluation metrics functions."""

    @classmethod
    def setUpClass(cls):
        """Create sample predictions for testing."""
        cls.y_test = pd.Series([1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        cls.y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 1])

    def test_accuracy_calculation(self):
        """Test train/test accuracy calculation."""
        y_train = pd.Series([1, 0, 1, 1, 0])
        y_train_pred = np.array([1, 0, 1, 0, 0])

        result = calculate_train_test_accuracy(
            y_train,
            y_train_pred,
            self.y_test,
            self.y_pred,
        )

        self.assertIn("train_accuracy", result)
        self.assertIn("test_accuracy", result)
        self.assertIn("accuracy_gap", result)
        self.assertTrue(0 <= result["train_accuracy"] <= 1)
        self.assertTrue(0 <= result["test_accuracy"] <= 1)

    def test_classification_metrics(self):
        """Test precision, recall, F1 calculation."""
        result = calculate_classification_metrics(self.y_test, self.y_pred)

        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1_score", result)
        self.assertIn("confusion_matrix", result)
        self.assertTrue(0 <= result["precision"] <= 1)
        self.assertTrue(0 <= result["recall"] <= 1)
        self.assertTrue(0 <= result["f1_score"] <= 1)


if __name__ == "__main__":
    unittest.main()
