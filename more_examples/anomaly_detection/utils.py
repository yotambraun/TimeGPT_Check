import pandas as pd

def load_and_prepare_data(file_path: str,number_of_rows_treshold:int=48) -> pd.DataFrame:
    """Loads the dataset and filters out groups with less than 48 rows."""
    df = pd.read_csv(file_path)
    df = df.groupby('unique_id').filter(lambda x: len(x) >= number_of_rows_treshold)
    return df