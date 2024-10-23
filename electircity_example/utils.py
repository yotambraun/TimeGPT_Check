import os
import pandas as pd
from typing import Tuple
from dotenv import load_dotenv
import matplotlib.pyplot as plt

def load_environment_variables() -> str:
    """Load environment variables from a .env file."""
    load_dotenv()
    return os.getenv("APIKEY_NIXTLA")

def create_directory(directory_path: str) -> None:
    """Create a directory if it doesn't already exist."""
    os.makedirs(directory_path, exist_ok=True)

def load_and_prepare_data(url: str) -> pd.DataFrame:
    """Load the dataset and prepare the datetime column."""
    df = pd.read_csv(url)
    df["ds"] = pd.to_datetime(df["ds"])
    return df

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into training and testing sets by unique_id.

    Returns:
        Tuple containing the training dataframe and the testing dataframe.
    """
    train_dfs, test_dfs = [], []
    for _, group in df.groupby('unique_id'):
        train, test = group[:-24], group[-24:]
        train_dfs.append(train)
        test_dfs.append(test)
    return pd.concat(train_dfs), pd.concat(test_dfs)

def plot_and_save_forecast(test_df: pd.DataFrame, save_dir: str) -> None:
    """Plot and save the actual vs forecasted values for each unique_id.

    Args:
        test_df: DataFrame containing the test data and forecasted values.
        save_dir: Directory path where plots should be saved.
    """
    for unique_id in test_df['unique_id'].unique():
        plt.figure(figsize=(10, 6))
        unique_data = test_df[test_df['unique_id'] == unique_id]
        plt.plot(unique_data['ds'], unique_data['y'], label='Actual data')
        plt.plot(unique_data['ds'], unique_data['TimeGPT'], label='Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'Forecast for the next 24 hours for {unique_id}')
        plt.legend()
        save_path = os.path.join(save_dir, f'forecast_{unique_id}.png')
        plt.savefig(save_path)
        plt.show()