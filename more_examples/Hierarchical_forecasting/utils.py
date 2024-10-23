import pandas as pd
import pickle as pkl
import numpy as np
from more_examples.Hierarchical_forecasting.reconciler_factory import ReconcilerFactory

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    df["ds"] = pd.to_datetime(df["ds"])
    return df

def load_pickle(file_path: str):
    """Load a pickled file."""
    with open(file_path, 'rb') as f:
        return pkl.load(f)

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (MAPE)."""
    if y_true == 0 and y_pred == 0:
        return 0
    elif y_true == 0:
        return 100
    else:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    

def get_reconcilers(selected_methods: list) -> list:
    """Return a list of reconciler objects based on the selected methods."""
    factory = ReconcilerFactory()
    reconcilers = []
    for method in selected_methods:
        reconcilers.append(factory.create_reconciler(method))
    return reconcilers