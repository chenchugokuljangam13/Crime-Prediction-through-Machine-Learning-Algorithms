from django.conf import settings
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools


class IBMFuturePrediction:
    df = ''

    def __init__(self):
        path = settings.MEDIA_ROOT + "\\" + "IBM.csv"
        self.df = pd.read_csv(path)

    def start_future_prediction(self):
        import datetime
        df = self.df[['Date', 'Volume']]

        df['Date'] = pd.to_datetime(df['Date'])
        # dp = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        dp = pd.to_datetime(df['Date'])
        # print(dp.min(), dp.max())
        df = df.groupby(dp)['Volume'].sum().reset_index()
        df = df.set_index('Date')
        df.index
        y = df['Volume'].resample('MS').mean()
        y['2018':]
        import itertools
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        print('Examples of parameter combinations for Seasonal ARIMA...')
        print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
        print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
        print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
        print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

        import statsmodels.api as sm
        import itertools
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(y,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit()
                # print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                except Exception as ex:
                    print("Exception is ", str(ex))
                    continue

        import statsmodels.api as sm
        mod = sm.tsa.statespace.SARIMAX(y,
                                        order=(1, 1, 1),
                                        seasonal_order=(1, 1, 0, 12),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        results = mod.fit()
        # print(type(results))
        pred_uc = results.get_forecast(steps=100)
        pred_ci = pred_uc.conf_int()

        ax = y.plot(label='observed', figsize=(14, 7))

        pred_uc.predicted_mean.plot(ax=ax, label='Future Forecast')

        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        plt.legend()
        # plt.show()
        return pred_ci
