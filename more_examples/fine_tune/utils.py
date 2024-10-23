import pandas as pd
import numpy as np

def load_and_prepare_data(path: str) -> pd.DataFrame:
    """Load the dataset and prepare the datetime column."""
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"])
    return df

def mape(y_true, y_pred):
    if y_true ==0 and y_pred == 0:
        return 0
    elif y_true == 0:
        return 100
    else:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100