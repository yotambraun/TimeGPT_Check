import os
import pandas as pd
from nixtla import NixtlaClient
from more_examples.config import DATASETTRAINPATH, DATASETTESTPATH
from more_examples.fine_tune.utils import load_and_prepare_data, mape
from more_examples.utils import load_environment_variables
from more_examples.fine_tune.config import DIRSAVEPLOTPATH


def fine_tune_forecast(client: NixtlaClient, df_train: pd.DataFrame, finetune_steps: int = 12, h: int = 1) -> pd.DataFrame:
    """Fine-tune a forecast model using Nixtla TimeGPT.

    Args:
        client: An instance of NixtlaClient.
        df_train: DataFrame containing training data.
        finetune_steps: Number of fine-tuning steps, default is 12.
        h: Forecast horizon, default is 1.

    Returns:
        DataFrame containing forecasted results.
    """
    return client.forecast(
        df=df_train,
        h=h,
        finetune_steps=finetune_steps,
        finetune_loss='mae',  # Using MAE as the loss function for fine-tuning
        time_col='ds',
        target_col='y',
    )


def calculate_mape_per_unique_id(df_test: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Merge forecast data into the test set and calculate MAPE for each unique_id.

    Args:
        df_test: DataFrame containing test data.
        forecast_df: DataFrame containing forecasted data.

    Returns:
        DataFrame with MAPE calculated for each unique_id.
    """
    df_test = pd.merge(df_test, forecast_df[["unique_id", "TimeGPT"]], on=['unique_id'], how='left')
    df_test["mape"] = df_test.apply(lambda x: mape(x["y"], x["TimeGPT"]), axis=1)

    # Create a results DataFrame to store MAPE for each unique_id
    df_res = pd.DataFrame(columns=['unique_id', 'mape'])
    
    for unique_id in df_test['unique_id'].unique():
        df_test_temp = df_test[df_test['unique_id'] == unique_id]
        mean_mape = df_test_temp['mape'].mean()
        df_res = pd.concat([df_res, pd.DataFrame({'unique_id': [unique_id], 'mape': [mean_mape]})], ignore_index=True)
    
    # Handle NaN values in MAPE by replacing with 100
    df_res["mape"] = df_res["mape"].fillna(100)
    return df_res


def main() -> None:
    """Main function to load data, fine-tune, calculate MAPE, and save results."""
    # Load environment variables and initialize NixtlaClient
    api_key = load_environment_variables()
    nixtla_client = NixtlaClient(api_key=api_key)
    # Load and prepare train and test datasets
    df_train = load_and_prepare_data(DATASETTRAINPATH)
    df_test = load_and_prepare_data(DATASETTESTPATH)

    # Filter out short time series
    df_train = df_train.groupby('unique_id').filter(lambda x: len(x) >= 36)

    # Fine-tune the forecast model
    timegpt_fcst_finetune_df = fine_tune_forecast(nixtla_client, df_train, finetune_steps=12, h=1)

    # Calculate MAPE per unique_id
    df_res = calculate_mape_per_unique_id(df_test, timegpt_fcst_finetune_df)

    # Save the results
    os.makedirs(DIRSAVEPLOTPATH, exist_ok=True)
    df_res.to_csv(os.path.join(DIRSAVEPLOTPATH, 'df_res.csv'), index=False)
    print(f"Results saved to {DIRSAVEPLOTPATH}")


if __name__ == "__main__":
    main()