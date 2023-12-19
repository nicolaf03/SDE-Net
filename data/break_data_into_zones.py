import pandas as pd
from pathlib import Path

ZONES = ["NORD", "CNOR", "CSUD", "SUD", "SICI", "SARD"]

curr_dir = Path(__file__).parent
file_path = curr_dir / 'wind_supply_ITA.csv'
df = pd.read_csv(file_path, delimiter=';', decimal=',', skiprows=2)

for i in range(len(ZONES)):
    zone = ZONES[i]
    sub_df = pd.DataFrame(df.iloc[:,2*i:(2*i+2)])
    sub_df.rename(columns={sub_df.columns[0]: 'Date', sub_df.columns[1]: 'Energy'}, inplace=True)
    sub_df.dropna(inplace=True)
    sub_df.iloc[:,0] = pd.to_datetime(sub_df.iloc[:,0], format="%d/%m/%Y")
    sub_df.set_index('Date', inplace=True)

    file_name = f'wind_{zone}.csv'
    sub_df.to_csv(curr_dir / file_name)
