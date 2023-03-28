from datetime import datetime, timedelta
import pandas as pd
import pytz


def is_start_solar_hour_day(date_str):
    return date_str in start_solar_hour_days()

def start_solar_hour_days():
    return ['2015-10-25', '2016-10-30', '2017-10-29', '2018-10-28', '2019-10-27', '2020-10-25']

def is_start_legal_hour_day(date_str):
    return date_str in start_legal_hour_days()

def start_legal_hour_days():
    # non inserisco '2020-03-29' perchè il file mensile è già corretto
    return ['2015-03-29', '2016-03-27', '2017-03-26', '2018-03-25', '2019-03-31']


def convert_from_datetime(datetime):
    date_str = datetime.strftime('%Y-%m-%d')
    hour_str = datetime.strftime('%H')
    hour = int(hour_str)
    hour_str = hour_to_str(hour)
    return date_str, hour_str


def convert_mensili_date_hour(date_str, hour_str):
    hour = int(hour_str)
    # hour conversion from 1-24 format to 0-23 format
    hour -= 1

    # hour is a value like 0 1 2 3 4 5 ... 23

    if is_start_solar_hour_day(date_str):
        # fix hour:  da 0 1 2 3 4 5 ... 24 -> 0 1 2 2 3 4 5 ... 23 (we gain one hour, but is missing in the csv!)
        if hour > 2:
            hour -= 1

    if is_start_legal_hour_day(date_str):
            # fix hour:  da 0 1 2 3 4 5 ... 22 -> 0 1 2 4 5 ... 23 (we loose one hour)
            if hour > 2:
                hour += 1 if hour!=24 else 0


    hour_str = hour_to_str(hour)
    date_hour_str = date_str + '-' + hour_str

    try:
        date = datetime.strptime(date_hour_str, '%Y-%m-%d-%H')
    except ValueError as e:
        print('datetime.strptime(date_hour_str) error, date_hour_str=', date_hour_str)
        raise

    tz = pytz.timezone('Europe/Rome')
    date = tz.localize(date)
    return date, date_hour_str


def create_datetime_sequence(start_date, end_date):
    tz = pytz.timezone('Europe/Rome')
    curr_date = start_date
    datetimes = dict()
    prev_hour_str = '0'
    while curr_date <= end_date:
        date_str, hour_str = convert_from_datetime(curr_date)
        if date_str not in datetimes:
            datetimes[date_str] = []

        if is_start_solar_hour_day(date_str) and hour_str == '02' and prev_hour_str != hour_str:
            # avoid hour duplicates in solar day
            prev_hour_str = hour_str
            curr_date = (curr_date + timedelta(hours=1)).astimezone(tz)
            continue

        datetimes[date_str].append((curr_date, hour_str))


        if is_start_legal_hour_day(date_str) and hour_str == '01':
            # add missing hour in legal day, copy the date from prev record
            datetimes[date_str].append((curr_date, '02'))

        prev_hour_str = hour_str
        curr_date = (curr_date + timedelta(hours=1)).astimezone(tz)

    return datetimes


def hour_to_str(hour):
    hour = int(hour)
    if hour < 9:
        hour = '0' + str(hour)

    return str(hour)


def convert_df_date_hour(df, hour_col='Ora', date_col='Data'):
    df[hour_col] = df[hour_col].apply(hour_to_str)
    date = pd.to_datetime(df[date_col] + '-' + df[hour_col], format='%Y-%m-%d-%H', exact=False)
    return date.dt.tz_localize('Europe/Rome', ambiguous=[False] * len(date), nonexistent=pd.Timedelta('-1H'))
