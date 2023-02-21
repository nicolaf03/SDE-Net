import pandas as pd
from pathlib import Path

ZONES = ["NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD"]

curr_dir = Path(__file__).parent
file_path = curr_dir / 'wind_supply_ITA.csv'
df = pd.read_csv(file_path, delimiter=';', decimal=',', skiprows=2)

for zone in ZONES:
    print(f'{zone}')
    
    sub_df = pd.DataFrame(df.iloc[:,0:2])
    sub_df.rename(columns={'Date': 'date', 'KWh (daily average)': 'energy'}, inplace=True)
    sub_df.dropna(inplace=True)
    sub_df.iloc[:,0] = pd.to_datetime(sub_df.iloc[:,0], format="%d/%m/%Y")
    sub_df.set_index('date', inplace=True)
    
    train = sub_df.loc[:'2021']
    test = sub_df.loc['2022']
    
    print(f"Train Set: {train.shape[0]} entries [{round(train.shape[0]/sub_df.shape[0],2)*100}%]")
    print(f" Test Set: {test.shape[0]} entries  [{round(test.shape[0]/sub_df.shape[0],2)*100}%]")

    sub_df.to_csv(curr_dir / f'wind_{zone}.csv')
    train.to_csv(curr_dir / f'wind_{zone}_train.csv')
    test.to_csv(curr_dir / f'wind_{zone}_test.csv')
