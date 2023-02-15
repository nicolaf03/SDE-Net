import pandas as pd
from pathlib import Path

ZONES = ["NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD"]

curr_dir = Path(__file__).parent


for zone in ZONES:
    file_path = curr_dir / f'wind_{zone}.csv'
    df = pd.read_csv(file_path)
    df['time_idx'] = df.index
    df.set_index('time_idx', inplace=True)
    
    file_name = curr_dir / f'wind_{zone}_idx.csv'
    df.to_csv(file_name)

print(0)