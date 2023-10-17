# Read electricity data and prepare a single csv file
# Operations being done:
# 1. Combine all years from 2013 to 2022
# 2. Remove duplicate records
import pandas as pd
import os
import zipfile


class DataPreparation:
    def __init__(self, data_folder_path):
        self.folder_path = data_folder_path

    def remove_dups(self, df):
        # Remove obvious duplicates at the record level
        unique_df = df.drop_duplicates()
        df_sorted = unique_df.sort_values(by=["zipcode", "year", "month"])
        df_first = df_sorted.groupby(["zipcode", "year", "month"]).first().reset_index()
        df_first['totalkWh'] = df_first['totalkWh'].str.replace(',', '').astype(int)
        return df_first['zipcode', 'year', 'month', 'customerclass', 'totalcustomers', 'totalkWh', 'averagekWh']

    def generate_writable_df(self, sub_folder_path):
        dataframes = []
        path = self.folder_path+sub_folder_path
        for directory_name in os.listdir(path):
            directory_path = os.path.join(path, directory_name)
            if directory_name.endswith('.zip'):
                with zipfile.ZipFile(directory_path, 'r') as zip_folder:
                    for filename in zip_folder.namelist():
                        if filename.endswith('.csv'):
                            df = pd.read_csv(zip_folder.open(filename), header=None)
                            dataframes.append(df.iloc[1:])

        if dataframes:
            df = pd.concat(dataframes, ignore_index=True)
        else:
            df = pd.DataFrame()
        df.columns = ['zipcode', 'month', 'year', 'customerclass', 'combined', 'totalcustomers', 'totalkWh', 'averagekWh']
        unique_df = self.remove_dups(df)
        return unique_df


dp_object = DataPreparation('datasets/')
dataframe_to_csv = dp_object.generate_writable_df('electricity_data/')
dataframe_to_csv.to_csv('datasets/processed/electricity_combined.csv', index=False)