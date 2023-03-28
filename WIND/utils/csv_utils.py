import pandas as pd
from pathlib import Path

curr_dir = Path(__file__).parent

def load_aggregated_data(file):
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index('date', inplace=True, drop=False)
    return df

