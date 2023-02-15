from pathlib import Path
import pandas as pd
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from torch.utils.data import DataLoader

curr_dir = Path(__file__).parent

def getDataSet(zone, batch_size, test_batch_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 0)
    kwargs.pop('input_size', None)
    print(f'Building {zone} data loader with {num_workers} workers')
    
    data_path = curr_dir / 'data' / f'wind_{zone}.csv'
    data = pd.read_csv(data_path)
    
    min_encoder_length = 0
    max_encoder_length = 400
    min_prediction_length = 1
    max_prediction_length = 7
    training_cutoff = data["time_idx"].max() - max_prediction_length
    
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx='time_idx',
        target='energy',
        group_ids=['energy'],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
        # target_normalizer=,
        allow_missing_timesteps=True,
        time_varying_unknown_reals=['energy']
    )
    
    testing = TimeSeriesDataSet.from_dataset(
        training, data, predict=True, stop_randomization=True
    )
    import pdb; pdb.set_trace()
    # create dataloaders for model
    train_loader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    test_loader = testing.to_dataloader(
        train=False, batch_size=test_batch_size, num_workers=0
    )
    
    #! it doesn't work
    # train_loader = DataLoader(
    #     training,
    #     batch_size=batch_size,
    #     shuffle=True, 
    #     num_workers=0
    # )
    # test_loader = DataLoader(
    #     testing,
    #     batch_size=test_batch_size,
    #     shuffle=False, 
    #     num_workers=0
    # )
    
    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = getDataSet(zone='SUD', batch_size=256, test_batch_size=1000)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        print(inputs.shape)
        print(targets.shape)
