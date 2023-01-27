import numpy as np
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.data import (
    TimeSeriesDataSet,
    GroupNormalizer
)

data = get_stallion_data()  # load data as pandas dataframe

# add time index
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()

# add additional features
# categories have to be strings
data["month"] = data["date"].dt.month.astype(str).astype("category")
data["log_volume"] = np.log(data.volume + 1e-8)
data["avg_volume_by_sku"] = (
    data
    .groupby(["time_idx", "sku"], observed=True)
    .volume.transform("mean")
)
data["avg_volume_by_agency"] = (
    data
    .groupby(["time_idx", "agency"], observed=True)
    .volume.transform("mean")
)

# we want to encode special days as one variable and 
# thus need to first reverse one-hot encoding
special_days = [
    "easter_day", "good_friday", "new_year", "christmas",
    "labor_day", "independence_day", "revolution_day_memorial",
    "regional_games", "fifa_u_17_world_cup", "football_gold_cup",
    "beer_capital", "music_fest"
]
data[special_days] = (
    data[special_days]
    .apply(lambda x: x.map({0: "-", 1: x.name}))
    .astype("category")
)

# show sample data
data.sample(10, random_state=521)

max_prediction_length = 6   # forecast 6 months
max_encoder_length = 24     # use 24 months of history
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=0,  # allow predictions without history
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=[
        "avg_population_2017",
        "avg_yearly_household_income_2017"
    ],
    time_varying_known_categoricals=["special_days", "month"],
    # group of categorical variables can be treated as one variable
    variable_groups={"special_days": special_days},
    time_varying_known_reals=[
        "time_idx",
        "price_regular",
        "discount_in_percent"
    ],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],
    target_normalizer=GroupNormalizer(
        groups=["agency", "sku"]#, coerce_positive=1.0
    ),  # use softplus with beta=1.0 and normalize by group
    add_relative_time_idx=True,     # add as feature
    add_target_scales=True,         # add as feature
    add_encoder_length=True,        # add as feature
)

# create validation set (predict=True) which means to predict the
# last max_prediction_length points in time for each series
validation = TimeSeriesDataSet.from_dataset(
    training, data, predict=True, stop_randomization=True
)

# create dataloaders for model
batch_size = 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0
)
