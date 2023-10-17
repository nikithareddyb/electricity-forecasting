import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
time_split_inner = TimeSeriesSplit(n_splits=3) # These splits are used to find the best GS hyper-parameters
import joblib


class ModelTesting:
    def __init__(self, data_path, model_path, method, model):
        self.source_path = data_path
        self.pkl_path = model_path
        self.mod = model
        self.method = method

    def get_testdata_ready(self):
        from sklearn.preprocessing import StandardScaler
        print('Reading a feature file for '+self.method)
        lagged_features = pd.read_csv(self.source_path, index_col=False)
        lagged_features = lagged_features.drop(columns=['datetime'])
        print(lagged_features.head())
        split = int(round(len(lagged_features) / 10, 0))
        train = lagged_features[:len(lagged_features) - split]
        test = lagged_features[len(lagged_features) - split:]
        std = StandardScaler()
        lagged_norm_train = std.fit_transform(train.iloc[:, :-1])
        lagged_norm_test = std.transform(test.iloc[:, :-1])
        train_target = np.array(train.iloc[:, -1]).reshape(-1, 1)
        test_target = np.array(test.iloc[:, -1]).reshape(-1, 1)
        std = StandardScaler()
        train_target_norm = std.fit_transform(train_target)
        test_target_norm = std.transform(test_target)
        return lagged_norm_test, test_target_norm

    def model_testing(self):
        lagged_norm_test, test_target_norm = self.get_testdata_ready()
        gs_las = joblib.load(self.pkl_path)
        y_pred = gs_las.predict(lagged_norm_test)
        y_true = test_target_norm

        from sklearn.metrics import mean_squared_error, mean_absolute_error
        # Mean Squared Error (MSE)
        # mse = mean_squared_error(actual_series, predicted_series)
        mse = mean_squared_error(y_true, y_pred)

        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)

        # Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_true, y_pred)

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print(self.mod+' Model for '+self.method+' method tested. Listed are the test results')
        print(f'MSE: {round(mse, 3)}\nRMSE: {round(rmse, 3)}\nMAE: {round(mae, 3)},\nMAPE: {round(mape, 3)}')
        # Define the mean and standard deviation used for standard normalization
        mean = np.mean(y_true)  # Replace with the actual mean value
        std = np.std(y_true)  # Replace with the actual standard deviation value

        # Apply inverse transformation (standardization)
        predicted_series = (y_pred * std) + mean
        actual_series = (y_true * std) + mean
        y_true = np.array(actual_series)
        y_pred = np.array(predicted_series)

        import matplotlib.pyplot as plt
        plt.plot(y_pred, label='Predictions')
        plt.plot(y_true, color='red', label='Actual')
        plt.legend()
        plt.title('Actual & Predicted Trend for Univariate Using Lasso')
        plt.savefig('models/testing_graphs/' + self.method + '_' + self.mod + '.jpg', bbox_inches='tight', format='jpg')
        plt.show()

method = 'exogenous'
model = 'MLP'
data_path = 'datasets/prepared/model-fitting/'+method+'_data_tofit.csv'
model_path = 'models/'+model+'_'+method+'.pkl'
model_test = ModelTesting(data_path, model_path, method, model)
model_test.model_testing()