import pandas as pd


def load_data(filepath):
    """
    Load the NDVI dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df


def get_ndvi_columns(df):
    """
    Return all NDVI time-series columns.
    """
    ndvi_cols = [
        col for col in df.columns
        if col.startswith("ndvi_")
    ]

    return ndvi_cols


def prepare_labels(df):
    """
    Encode land-cover labels into numerical classes.
    """
    label_mapping = {
        "deforested": 0,
        "forest": 1,
        "old_clearing": 2
    }

    y = df["label"].map(label_mapping)

    return y