from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_true, y_pred, class_names):
    """
    Calculate classification metrics and return them as a dictionary.
    """

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    }

    print("Accuracy:", results["accuracy"])
    print("Macro Precision:", results["macro_precision"])
    print("Macro Recall:", results["macro_recall"])
    print("Macro F1:", results["macro_f1"])

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

    return results


def get_confusion_matrix(y_true, y_pred):
    """
    Return the confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)