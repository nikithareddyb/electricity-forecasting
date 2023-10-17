import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

class Univariate_Time_Series_By_ZipCode:

	def __init__(self, period):
		self.__period = period
		self.__scaler = MinMaxScaler()

	def __get_unique_zips(self, dataframe, zip_name):

		# Extract the column with zip codes from the DataFrame
		zip_column = dataframe[zip_name].astype(str)
		# Split each string in the column into a list of zip codes
		zip_lists = [s.split(',') for s in zip_column]
		# Join the resulting lists together into a single list
		zip_list = list(itertools.chain.from_iterable(zip_lists))
		# Get the unique zip codes from the list
		return set(zip_list)

	def ____process_zip(self, dataframe, zip, zip_name, target_name):
		elec_zip = dataframe[(dataframe[zip_name] == int(zip))]
		elec_zip['timestamp'] = pd.to_datetime(elec_zip['Year'].astype(str) + '-' + elec_zip['Month'].astype(str), format='%Y-%m')
		elec_zip.sort_values('timestamp', ascending=True, inplace=True)
		elec_zip.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
		elec_zip.reset_index(drop=True, inplace=True)
		if (len(elec_zip) > 100):
			period = self.__period
			train_arr = np.zeros((len(elec_zip)-period, period))
			test_arr = []
			count = 0
			for i in range(period, len(elec_zip)):
				l = []
				for j in range(i-period, i):
					l.append(elec_zip[target_name][j])
				train_arr[count] = l
				count+=1
				test_arr.append(elec_zip[target_name][i])
			colnames = [target_name+'_t-6', target_name+'_t-5', target_name+'_t-4', 
						target_name+'_t-3', target_name+'_t-2', target_name+'_t-1']
			p_df = pd.DataFrame(train_arr, columns=colnames)
			p_df[zip_name] = zip
			p_df[target_name] = test_arr
			return p_df
		return pd.DataFrame()

	def __scale_dataframe(self, dataframe):
		return pd.DataFrame(self.__scaler.fit_transform(dataframe), columns=dataframe.columns)


	def fit_transform(self, dataframe, zip_name, target_name):

		if (("Year" not in dataframe.columns) or
		   ("Month" not in dataframe.columns) or
		   (zip_name not in dataframe.columns) or
		   (target_name not in dataframe.columns)):
			print("DataFrame should have columns: Month, Year, " 
		   				  + zip_name + ', ' + target_name)
			raise KeyError

		final_df = pd.DataFrame()
		unique_zips = self.__get_unique_zips(dataframe, zip_name)
		for zip in list(unique_zips):
			df = self.____process_zip(dataframe, zip, zip_name, target_name)
			final_df = pd.concat([final_df, df], ignore_index=True)
		return self.__scale_dataframe(final_df)

class LSTM_model():

	def __init__(self, input_shape):
		model = Sequential()
		model.add(LSTM(50, activation='relu', input_shape=input_shape))
		model.add(Dense(1024, activation='relu'))
		model.add(Dense(512, activation='relu'))
		model.add(Dense(1))
		model.compile(optimizer='adam', loss='mse')
		self.__model = model

	def compile(self, optimizer, loss):
		self.__model.compile(optimizer=optimizer, loss=loss)

	def fit(self, *args, **kwargs):
		self.__model.fit(*args, **kwargs)

	def evaluate(self, X, Y):
		self.__model.evaluate(X, Y)

	def save(self, filename):
		self.__model.save(filename)

def main():
	df=pd.read_csv('Electricity_v1_rb.csv')
	df = df[df['CustomerClass'] == 'Elec- Residential']

	time_series_processor = Univariate_Time_Series_By_ZipCode(period=6)
	print("Extracting Time Series")
	df = time_series_processor.fit_transform(df, 'Zip Code', 'TotalkWh')

	df_train, df_test = train_test_split(df, test_size=0.25, shuffle=True)
	y_train = df_train['TotalkWh'].to_numpy()
	y_test = df_test['TotalkWh'].to_numpy()
	X_train = np.asarray(df_train.drop(['TotalkWh'],axis=1).to_numpy(), dtype=float)
	X_test = np.asarray(df_test.drop(['TotalkWh'],axis=1).to_numpy(), dtype=float)
	df_test.to_csv('LSTM_test.csv', index=False)

	print("Creating Model")
	model = LSTM_model((7,1))
	model.compile('adam', 'mse')

	print("Training Model")
	model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1)

	model.save('electricity_LSTM.h5')
	print("Model Saved Successfully")


if __name__ == "__main__":
    main()
