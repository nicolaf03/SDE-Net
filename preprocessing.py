import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import periodogram


if __name__=='__main__':
    
    # import dataset SUD
    zone = 'SUD'
    PATH = f'data/wind_{zone}.csv'
    data = pd.read_csv(PATH)
    
    # convert date to format datetime
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])

    # complete all the datetimes
    all_datetimes = pd.DataFrame(pd.date_range(datetime(df.date.min().year, 1, 1), df.date.max(), freq='d'), columns=["date"])
    df = all_datetimes.merge(df, on=['date'], how='outer')
    df.set_index('date', inplace=True)

    # fill the missing values with the closest available data
    df.energy = df.energy.fillna(method='ffill').fillna(method='bfill')

    # check if every year has 365 data points (except the last year)
    df = df.reset_index(drop=False)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df.set_index('date', inplace=True)

    count_year_df = df.groupby('year')['energy'].count()
    print(count_year_df)
    
    # use only complete years
    condition = count_year_df >= 365
    complete_years = count_year_df.loc[condition].index
    df = df.loc[df.year.isin(complete_years)]

    df.energy.plot(figsize=(20,8))
    plt.title('Energy')
    plt.show()

    #
    # Time series decomposition
    
    # take logarithm
    dflog = np.log(df.loc[df.year < 2022, 'energy'])

    res = seasonal_decompose(dflog, model='additive', period=365)
    fig = res.plot()
    fig.set_size_inches((18,12))
    plt.show()


    # smooth the trend
    x_smooth = res.trend.rolling(150, min_periods=1, center=True).mean()

    plt.figure(figsize=(20,8))
    plt.plot(res.trend, label='trend')
    plt.plot(x_smooth, label='filtered')
    plt.legend()
    plt.show()
    
    
    # recover missing data by forward and backward fill
    trend = x_smooth
    trend = trend.fillna(method='ffill').fillna(method='bfill')

    plt.figure(figsize=(20,8))
    dflog.plot(label='log energy')
    trend.plot(label='trend', linewidth=3)
    plt.legend()
    plt.show()
    
    
    # obtain the detrended series
    detrended = dflog - trend
    detrended.plot(figsize=(20,8))
    plt.show()


    # plot periodogram
    f, Pxx_den = periodogram(detrended)
    fig = plt.figure(figsize=(20,8))
    fig = plt.plot(f,Pxx_den)
    plt.show()

    # get the highest frequency
    freq = f[Pxx_den > 120]

    # which correspond to a certain period
    T = int(1/freq)
    print(T)


    # deseasonalize the series
    # Perform Fourier Transform
    fft_result = np.fft.rfft(detrended)
    frequencies = np.fft.rfftfreq(len(detrended), d=1)

    # Identify frequencies corresponding to the seasonality and set them to zero
    seasonal_freq = 1 / T
    fft_result[np.abs(frequencies - seasonal_freq) < 1e-3] = 0

    # Perform Inverse Fourier Transform
    deseasonalized = np.fft.irfft(fft_result, n=len(detrended))

    deseasonalized_series = pd.Series(deseasonalized, index=detrended.index)
    deseasonalized_series.plot(figsize=(20,8))
    plt.show()
    
    
    # check for other periodicities
    # plot periodogram
    f, Pxx_den = periodogram(deseasonalized_series)
    fig = plt.figure(figsize=(20,8))
    fig = plt.plot(f,Pxx_den)
    plt.show()

    # get the highest frequency
    freq = f[Pxx_den > 20]

    # which correspond to a certain period
    T = [int(x) for x in 1/freq]
    print(T)
    
    # no more relevant periodicities