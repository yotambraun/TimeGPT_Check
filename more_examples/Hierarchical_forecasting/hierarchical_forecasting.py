import pandas as pd
from nixtla import NixtlaClient
from hierarchicalforecast.core import HierarchicalReconciliation
from more_examples.Hierarchical_forecasting.utils import load_data, load_pickle, mape, get_reconcilers
from more_examples.Hierarchical_forecasting.config import DATASETTRAINAGGPATH, DATASETTESTPATH, DATASETSDFPATH, TAGSPICKELPATH
from more_examples.utils import load_environment_variables


def perform_forecast(client: NixtlaClient, df_train: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    """Perform forecast using Nixtla TimeGPT."""
    timegpt_fcst = client.forecast(df=df_train, h=h, freq='ME', add_history=True)
    timegpt_fcst["ds"] = pd.to_datetime(timegpt_fcst["ds"])
    return timegpt_fcst


def reconcile_forecasts(timegpt_fcst: pd.DataFrame, df_train: pd.DataFrame, s_df: pd.DataFrame, tags: dict, selected_methods: list) -> pd.DataFrame:
    """Reconcile hierarchical forecasts using selected methods."""
    
    reconcilers = get_reconcilers(selected_methods)
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    
    timegpt_fcst = timegpt_fcst.set_index('unique_id')
    df_train = df_train.set_index('unique_id')

    Y_rec_df = hrec.reconcile(
        Y_hat_df=timegpt_fcst, 
        Y_df=df_train, 
        S=s_df, 
        tags=tags
    )
    return Y_rec_df.reset_index()


def evaluate_forecasts(Y_rec_df: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """Evaluate forecasts by calculating MAPE."""
    Y_rec_df = pd.merge(Y_rec_df, df_test, on=['unique_id'], how='left')
    
    models_names = Y_rec_df.drop(columns=["unique_id", "ds", "y"]).columns
    for model in models_names:
        Y_rec_df["mape"] = Y_rec_df.apply(lambda x: mape(x["y"], x[model]), axis=1)
        print(f"Mean MAPE for {model}: {Y_rec_df['mape'].mean()}")

    for unique_id in Y_rec_df['unique_id'].unique():
        print(f"Unique ID: {unique_id}")
        for model in models_names:
            mape_value = Y_rec_df[(Y_rec_df['unique_id'] == unique_id) & (Y_rec_df[model] != 0)]['mape'].mean()
            print(f"Model: {model}, MAPE: {mape_value}")
        print("*********************************************************************")


def main() -> None:
    """Main function to perform hierarchical forecasting."""
    api_key = load_environment_variables()
    nixtla_client = NixtlaClient(api_key=api_key)

    # Load datasets
    df_train_agg = load_data(DATASETTRAINAGGPATH)
    df_test = load_data(DATASETTESTPATH)
    s_df = load_data(DATASETSDFPATH)
    tags = load_pickle(TAGSPICKELPATH)

    # Select reconciliation methods dynamically
    selected_methods = [
        'MinTraceOLS', 
        'MinTraceShrink', 
        'BottomUp', 
        'ERMClosed', 
        'OptimalCombinationWLSStruct'
    ]

    # Perform forecast
    timegpt_fcst = perform_forecast(nixtla_client, df_train_agg)

    # Reconcile forecasts
    Y_rec_df = reconcile_forecasts(timegpt_fcst, df_train_agg, s_df, tags, selected_methods)

    # Filter reconciliation output and evaluate
    Y_rec_df = Y_rec_df.drop(columns=["y"])

    # Evaluate the forecasts
    evaluate_forecasts(Y_rec_df, df_test)


if __name__ == "__main__":
    main()