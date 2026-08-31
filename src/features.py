import numpy as np
import pandas as pd


def create_features(df, ndvi_cols):
    """
    Create statistical and temporal features from NDVI time series.
    """

    ndvi = df[ndvi_cols]

    features = pd.DataFrame(index=df.index)

    # Statistical features
    features["ndvi_mean"] = ndvi.mean(axis=1)
    features["ndvi_std"] = ndvi.std(axis=1)
    features["ndvi_min"] = ndvi.min(axis=1)
    features["ndvi_max"] = ndvi.max(axis=1)
    features["ndvi_range"] = (
        features["ndvi_max"] - features["ndvi_min"]
    )

    # Temporal features
    features["ndvi_first"] = ndvi.iloc[:, 0]
    features["ndvi_last"] = ndvi.iloc[:, -1]

    features["overall_change"] = (
        features["ndvi_last"] - features["ndvi_first"]
    )

    # Largest consecutive NDVI decrease
    differences = ndvi.diff(axis=1)

    features["largest_drop"] = differences.min(axis=1)

    # Linear trend
    time_steps = np.arange(len(ndvi_cols))

    features["ndvi_slope"] = ndvi.apply(
        lambda row: np.polyfit(time_steps, row.values, 1)[0],
        axis=1
    )

    # Early and late NDVI means
    midpoint = len(ndvi_cols) // 2

    features["early_mean"] = (
        ndvi.iloc[:, :midpoint].mean(axis=1)
    )

    features["late_mean"] = (
        ndvi.iloc[:, midpoint:].mean(axis=1)
    )

    features["early_late_change"] = (
        features["late_mean"] - features["early_mean"]
    )

    # Time step at which minimum NDVI occurs
    features["min_ndvi_timestep"] = (
        ndvi.values.argmin(axis=1)
    )

    return features