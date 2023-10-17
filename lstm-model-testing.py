import pandas as pd
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

def main():

	df_test = pd.read_csv('LSTM_test.csv')
	y_test = df_test['TotalkWh'].to_numpy()
	X_test = np.asarray(df_test.drop(['TotalkWh'],axis=1).to_numpy(), dtype=float)

	print("Loading Model")
	model = load_model('electricity_LSTM.h5')
	model.evaluate(X_test, y_test)


if __name__ == "__main__":
    main()