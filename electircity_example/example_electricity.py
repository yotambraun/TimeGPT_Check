import pandas as pd
from nixtla import NixtlaClient
from config import DATASETURL, DIRSAVEPLOTPATH
from utils import load_environment_variables, create_directory, load_and_prepare_data, split_data, plot_and_save_forecast


def forecast_with_nixtla(client: NixtlaClient, train_df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Generate forecast using the Nixtla TimeGPT API.

    Args:
        client: An instance of NixtlaClient.
        train_df: DataFrame with the training data.
        horizon: The forecast horizon, default is 24.

    Returns:
        DataFrame with forecasted values.
    """
    return client.forecast(train_df, h=horizon)


def merge_forecast_with_test(fcst_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Merge forecast results with the test dataframe.

    Args:
        fcst_df: DataFrame with forecasted data.
        test_df: DataFrame with actual test data.

    Returns:
        DataFrame that combines both test data and forecasted data.
    """
    test_df = test_df.reset_index(drop=True)
    fcst_df = fcst_df.reset_index(drop=True)
    test_df["TimeGPT"] = fcst_df["TimeGPT"]
    return test_df


def main() -> None:
    """Main function to execute the workflow."""
    # Load API Key and set up paths
    api_key = load_environment_variables()
    nixtla_client = NixtlaClient(api_key=api_key)

    create_directory(DIRSAVEPLOTPATH)

    # Load and prepare data
    df = load_and_prepare_data(DATASETURL)

    # Split the data into train and test sets
    train_df, test_df = split_data(df)
    print(f"Training set: {train_df.shape[0]} samples")
    print(f"Test set: {test_df.shape[0]} samples")

    # Perform forecasting
    fcst_df = forecast_with_nixtla(nixtla_client, train_df, horizon=24)

    # Merge forecast with test set
    test_df = merge_forecast_with_test(fcst_df, test_df)
    print(f"Final test_df: {test_df.head()}")

    # Plot and save forecast
    plot_and_save_forecast(test_df, DIRSAVEPLOTPATH)


if __name__ == "__main__":
    main()