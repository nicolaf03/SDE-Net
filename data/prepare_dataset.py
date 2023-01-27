import pandas as pd
from pathlib import Path

ZONES = ["NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD"]

curr_dir = Path(__file__).parent


for zone in ['SUD']:
    file_path = curr_dir / f'wind_{zone}.csv'
    df = pd.read_csv(file_path)
    df['time_idx'] = df.index

print(0)