from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)


def create_random_forest():
    """
    Create the baseline Random Forest classifier.
    """
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )


def create_extra_trees():
    """
    Create the Extra Trees classifier.
    """
    return ExtraTreesClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )