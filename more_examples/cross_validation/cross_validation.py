import os
import pandas as pd
import matplotlib.pyplot as plt
from nixtla import NixtlaClient
from more_examples.cross_validation.utils import load_and_prepare_data, mape
from more_examples.utils import load_environment_variables
from more_examples.config import DATASETTRAINPATH
from more_examples.cross_validation.config import DIRSAVEPLOTPATH


def perform_cross_validation(client: NixtlaClient, df_train: pd.DataFrame, h: int = 1, n_windows: int = 12) -> pd.DataFrame:
    """Perform cross-validation using Nixtla's TimeGPT API.

    Args:
        client: An instance of NixtlaClient.
        df_train: Training DataFrame.
        h: Forecast horizon, default is 1.
        n_windows: Number of cross-validation windows, default is 12.

    Returns:
        DataFrame with cross-validation results.
    """
    cv_df = client.cross_validation(
        df=df_train,
        h=h,
        n_windows=n_windows,
        time_col='ds',
        target_col='y',
        freq='ME'
    )
    cv_df["ds"] = pd.to_datetime(cv_df["ds"])
    return cv_df


def calculate_mape(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate MAPE (Mean Absolute Percentage Error) for each row in the DataFrame.

    Args:
        cv_df: DataFrame containing actual and forecasted values.

    Returns:
        DataFrame with MAPE calculated for each row.
    """
    cv_df["mape"] = cv_df.apply(lambda x: mape(x["y"], x["TimeGPT"]), axis=1)
    return cv_df


def plot_and_save_cross_validation(cv_df: pd.DataFrame, save_dir: str) -> None:
    """Plot and save the cross-validation forecast results for each unique_id.

    Args:
        cv_df: DataFrame containing cross-validation results.
        save_dir: Directory path to save the plots.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for unique_id in cv_df['unique_id'].unique():
        mean_mape = cv_df[cv_df['unique_id'] == unique_id]['mape'].mean()
        plt.figure(figsize=(10, 6))
        plt.plot(cv_df[cv_df['unique_id'] == unique_id]['ds'], cv_df[cv_df['unique_id'] == unique_id]['y'], label='Actual data')
        plt.plot(cv_df[cv_df['unique_id'] == unique_id]['ds'], cv_df[cv_df['unique_id'] == unique_id]['TimeGPT'], label='Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'Forecast for the 12 cross-val windows for {unique_id} and mean MAPE: {mean_mape}')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'forecast_{unique_id}.png'))
        plt.close()


def main() -> None:
    """Main function to perform cross-validation, calculate MAPE, and visualize results."""
    # Load API key and initialize NixtlaClient
    api_key = load_environment_variables()
    nixtla_client = NixtlaClient(api_key=api_key)

    # Load and prepare data
    df_train = load_and_prepare_data(DATASETTRAINPATH)

    # Perform cross-validation
    imegpt_cv_df = perform_cross_validation(nixtla_client, df_train, h=1, n_windows=12)

    # Calculate MAPE
    imegpt_cv_df = calculate_mape(imegpt_cv_df)

    # Plot and save cross-validation results
    plot_and_save_cross_validation(imegpt_cv_df, DIRSAVEPLOTPATH)


if __name__ == "__main__":
    main()