from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


class StationarityCheck:
    def __init__(self, data, col):
        self.uni_data = data
        self.column = col

    def stationary_check_elec(self, check_data):
        # Lower p-value implies stationarity
        df = adfuller(check_data[self.column], autolag='AIC')
        return df[1]

    def extract_compts_elec(self):
        p_val = self.stationary_check_elec(self.uni_data)
        print("P-value of the original data is:", p_val)

        self.result = seasonal_decompose(self.uni_data[self.column], model='additive', period=12)

        # Increase the figure size
        plt.figure(figsize=(12, 8))  # Adjust the figsize as desired

        # Plot the original time series
        plt.subplot(411)
        self.uni_data[self.column].plot()
        plt.title("Original Data")

        # Plot the trend component
        plt.subplot(412)
        self.result.trend.plot()
        plt.title("Trend Component")

        # Plot the seasonality component
        plt.subplot(413)
        self.result.seasonal.plot()
        plt.title("Seasonal Component")

        # Plot the residual component
        plt.subplot(414)
        self.result.resid.plot()
        plt.title("Residual Component - Stationary")

        plt.tight_layout()
        plt.savefig('datasets/prepared/graphs/timeseries_plots.jpg', bbox_inches='tight', format='jpg')
        print("The graph in the window shows various timeseries plots on TotalkW""h with variation indicated in the title of the plot")
        print('***After taking a look at the graph please close the window to proceed with the further code.')
        plt.show()

    def post_stat_create(self):
        # Extracting residual component as the
        residuals = self.result.resid

        # Increase the figure size
        plt.figure(figsize=(12, 7))
        plt.title('Stationary Electricity Consumption Data')

        residuals.plot()
        plt.savefig('datasets/prepared/graphs/Stationarized-Electricity-Consumption-Timeseries.jpg', bbox_inches='tight',
                    format='jpg')
        print("The graph in the window shows timeseries plots on TotalkWh after Stationarity adjustment")
        print('***After taking a look at the graph please close the window to proceed with the further code.')
        plt.show()

        lag = int(pd.DataFrame(residuals).isna().sum())
        stationary_data = pd.DataFrame(residuals).dropna()
        stationary_data.index = self.uni_data['datetime'][:len(self.uni_data) - lag]
        stationary_data.rename(columns={'resid': self.column}, inplace=True)
        p_val = self.stationary_check_elec(stationary_data)
        print("P-value: ", p_val)
        return stationary_data


column_name = 'totalcustomers' #'totalkWh'
source_path = "datasets/prepared/bivariate_cluster_1.csv"
bivariate_data = pd.read_csv(source_path)
#bivariate_data['totalkWh'] = bivariate_data['totalkWh'].str.replace(',', '').astype(int)

stationary = StationarityCheck(bivariate_data, column_name)
stationary.extract_compts_elec()
stationary_data = stationary.post_stat_create()
stationary_data.to_csv('datasets/prepared/model-fitting/stationary_'+column_name+'_electricity.csv')