import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def calculate_train_test_accuracy(
    y_train: pd.Series,
    y_train_pred: object,
    y_test: pd.Series,
    y_test_pred: object,
) -> dict[str, float]:
    """Calculate train/test accuracy and the generalization gap."""
    train_accuracy = float(accuracy_score(y_train, y_train_pred))
    test_accuracy = float(accuracy_score(y_test, y_test_pred))
    accuracy_gap = train_accuracy - test_accuracy

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "accuracy_gap": accuracy_gap,
    }


def calculate_classification_metrics(
    y_true: pd.Series,
    y_pred: object,
) -> dict[str, object]:
    """Calculate precision, recall, F1, and confusion matrix details."""
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        confusion_payload: dict[str, object] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
    else:
        confusion_payload = {"matrix": cm.tolist()}

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": confusion_payload,
    }


def build_classification_metrics_response(
    *,
    message: str,
    model_name: str,
    y_train: pd.Series,
    y_train_pred: object,
    y_test: pd.Series,
    y_test_pred: object,
) -> dict[str, object]:
    """Build a standard response payload for classification endpoints."""
    response = {
        "message": message,
        "model": model_name,
        "actual_values": y_test.iloc[:10].tolist(),
        "predicted_values": y_test_pred[:10].tolist(),
    }
    response.update(calculate_train_test_accuracy(y_train, y_train_pred, y_test, y_test_pred))
    response.update(calculate_classification_metrics(y_test, y_test_pred))
    return response
