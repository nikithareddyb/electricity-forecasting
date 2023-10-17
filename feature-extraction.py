import pandas as pd
import datetime
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


class FeatureExtraction:
    def __init__(self, path1, path2, path3):
        self.totalkWh_path = path1
        self.totalcustomers_path = path2
        self.exogenous_path = path3
        self.num_lags_totalkWh = 12
        self.num_lags_totalCustomer = 2

        # Create a lagged version of the time series data
    def create_lagged_features(self, data, num_lags):
        lagged_data = pd.DataFrame()
        for i in range(1, num_lags + 1):
            lagged_data[f'Lag{i}'] = data.shift(i)
        return lagged_data

    def extract_features(self, typ):
        stat_elec_1 = pd.read_csv(self.totalkWh_path, index_col='datetime')
        # Generate some sample time series data
        totalkWh_data = pd.Series(stat_elec_1['totalkWh'])
        # Create lagged features and Remove rows with missing values (due to shifting)
        self.lagged_features_totalkWh_data = self.create_lagged_features(totalkWh_data, self.num_lags_totalkWh).dropna()
        self.uni_lagged_features = self.lagged_features_totalkWh_data
        # Add the target variable (the next value in the time series)
        self.uni_lagged_features['Target'] = totalkWh_data[self.num_lags_totalkWh:]

        stat_elec_2 = pd.read_csv(self.totalcustomers_path, index_col='datetime')
        # Create a lagged version of the time series data
        totalCustomer_data = pd.Series(stat_elec_2['totalcustomers'])
        # Create lagged features for customers
        self.lagged_features_totalcustomers_data = self.create_lagged_features(totalCustomer_data, self.num_lags_totalCustomer).dropna()
        # Merge lagged features for customers and electricity
        self.lagged_features_bi = pd.merge(self.lagged_features_totalcustomers_data, self.uni_lagged_features,
                                           on='datetime', how='inner')

        if typ == 'univariate':
            print('preparing univariate')
            to_write_df = self.uni_lagged_features
            print('finished assignment')
        elif typ == 'bivariate':
            print('preparing bivariate')
            to_write_df = self.lagged_features_bi
            print('finished assignment')
        elif typ == 'exogenous':
            print('preparing exogenous')
            exogenous = pd.read_csv(self.exogenous_path)
            self.lagged_features_bi.reset_index(inplace=True)
            # Extract the year
            self.lagged_features_bi['Year'] = pd.to_datetime(self.lagged_features_bi['datetime']).dt.year
            exogenous['Year'] = pd.to_datetime(exogenous['datetime']).dt.year
            lagged_features_exo = pd.merge(self.lagged_features_bi, exogenous, on='datetime', how='left')
            dummies = pd.get_dummies(lagged_features_exo['season']).drop('Winter', axis=1)
            lagged_features_exo['Fall'] = dummies['Fall']
            lagged_features_exo['Spring'] = dummies['Spring']
            lagged_features_exo['Summer'] = dummies['Summer']
            lagged_features_exo.index = lagged_features_exo['datetime']
            lagged_features_exo.drop('datetime', axis=1, inplace=True)
            lagged_features_exo.drop('season', axis=1, inplace=True)
            lagged_features_exo = lagged_features_exo[
                ['Lag1_x', 'Lag2_x', 'Lag1_y', 'Lag2_y', 'Lag3', 'Lag4', 'Lag5', 'Lag6', 'Lag7', 'Lag8', 'Lag9',
                 'Lag10', 'weekends', 'Fall', 'Spring', 'Summer', 'Target']]
            to_write_df = lagged_features_exo
            print('finished assignment')
        return to_write_df


method = 'bivariate'

source_path_totalkWh = 'datasets/prepared/model-fitting/stationary_totalkWh_electricity.csv'
source_path_totalcustomers = 'datasets/prepared/model-fitting/stationary_totalcustomers_electricity.csv'
source_path_exo = 'datasets/prepared/exogenous_cluster_1.csv'

feature_extractor = FeatureExtraction(source_path_totalkWh, source_path_totalcustomers, source_path_exo)

df_writable = feature_extractor.extract_features(method)
df_writable.to_csv('datasets/prepared/model-fitting/'+method+'_data_tofit.csv')
