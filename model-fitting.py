import pandas as pd
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
import sys


# These splits are used to find the best GS hyper-parameters
time_split_inner = TimeSeriesSplit(n_splits=3)

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


class ModelTraining:
    def __init__(self, path, method, model):
        self.source_path = path
        self.mod = model
        self.method = method

    def get_data_ready(self):
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
        return lagged_norm_train, lagged_norm_test, train_target_norm, test_target_norm

    def model_fitting(self):
        if self.mod == 'Lasso':
            lagged_norm_train, lagged_norm_test, train_target_norm, test_target_norm = self.get_data_ready()

            from sklearn.linear_model import Lasso
            lasso = Lasso(fit_intercept=1, alpha=0.05, max_iter=10000, random_state=8)

            las_params = {'fit_intercept': [1, 0],
                          'alpha': [0.005, 0.01, 0.03, 0.05, 0.07, 0.1]}
            gs_las = GridSearchCV(lasso, las_params, cv=time_split_inner, scoring='neg_mean_squared_error',
                                  n_jobs=-1, verbose=1)

            gs_las.fit(lagged_norm_train, train_target_norm.ravel())
            gs_las.best_params_
            print('Model Lasso for '+self.method+' prediction trained!')
            return gs_las

        if self.mod == 'MLP':
            lagged_norm_train, lagged_norm_test, train_target_norm, test_target_norm = self.get_data_ready()
            mlp = MLPRegressor(hidden_layer_sizes=(24,), alpha=1e-6, activation='relu', early_stopping=True,
                               max_iter=20000, random_state=8)

            mlp_params = {'hidden_layer_sizes': [(24,), (36,), (24, 24), (24, 12), (36, 24), (36, 12)],
                          'alpha': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
                          'activation': ['relu', 'identity', 'tanh']}

            gs_mlp = GridSearchCV(mlp, mlp_params, cv=time_split_inner, scoring='neg_mean_squared_error',
                                  n_jobs=-1, verbose=1)

            gs_mlp.fit(lagged_norm_train, train_target_norm.ravel())
            gs_mlp.best_params_
            print('Model MLP for '+self.method+' prediction trained!')
            return gs_mlp

        if self.mod == 'SVR':
            lagged_norm_train, lagged_norm_test, train_target_norm, test_target_norm = self.get_data_ready()
            svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': [1e-3, 1e-2, 1e-1, 0.1],
                'kernel': ['linear', 'rbf']
            }

            gs_svr = GridSearchCV(svr, param_grid, cv=time_split_inner, scoring='neg_mean_squared_error',
                                  n_jobs=-1, verbose=1)

            gs_svr.fit(lagged_norm_train, train_target_norm.ravel())
            gs_svr.best_params_
            print('Model SVR for '+self.method+' prediction trained!')
            return gs_svr


method = 'exogenous'
model = 'MLP'
data_path = 'datasets/prepared/model-fitting/'+method+'_data_tofit.csv'
file_path = 'models/'+model+'_'+method+'.pkl'
model_train = ModelTraining(data_path, method, model)
trained_mod = model_train.model_fitting()
joblib.dump(trained_mod, file_path)