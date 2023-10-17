# This python program will generate data for univariate, bivariate and exogenous modeling

import pandas as pd
import datetime
import random


class DataPreparation:
    def __init__(self, preparation_type):
        self.type = preparation_type

    def get_weekdays(self, year, month):
        # get the number of days in the month
        if month + 1 == 13:
            num_days = (datetime.date(year + 1, 1, 1) - datetime.date(year, month, 1)).days
        else:
            num_days = (datetime.date(year, month + 1, 1) - datetime.date(year, month, 1)).days
        # initialize counters for weekdays and weekends
        num_weekdays = 0
        num_weekends = 0
        # iterate over all days in the month
        for day in range(1, num_days + 1):
            # create a date object for the day
            date = datetime.date(year, month, day)
            # check if the day is a weekday (0-4) or weekend (5-6)
            if date.weekday() < 5:
                num_weekdays += 1
            else:
                num_weekends += 1
        return num_weekdays

    def get_weekends(self, year, month):
        # get the number of days in the month
        if month + 1 == 13:
            num_days = (datetime.date(year + 1, 1, 1) - datetime.date(year, month, 1)).days
        else:
            num_days = (datetime.date(year, month + 1, 1) - datetime.date(year, month, 1)).days
        # initialize counters for weekdays and weekends
        num_weekdays = 0
        num_weekends = 0
        # iterate over all days in the month
        for day in range(1, num_days + 1):
            # create a date object for the day
            date = datetime.date(year, month, day)
            # check if the day is a weekday (0-4) or weekend (5-6)
            if date.weekday() < 5:
                num_weekdays += 1
            else:
                num_weekends += 1
        return num_weekends

    def pick_zipcode_from_cluster(self, cluster_path):
        print('Cluster path', cluster_path)
        with open(cluster_path, 'r') as file:
            content = file.read()
        # Split the content using a comma as the delimiter
        values = content.split(',')
        # Trim leading and trailing whitespaces from each value
        zipcodes = [value.strip() for value in values]
        # extract the count of records for each zipcode in electricity_combined.csv
        elec_df = pd.read_csv(electricity_processed_path)
        useable_zips = []
        for i in zipcodes:
            zip_df = elec_df[elec_df['zipcode'] == int(i)]
            count = len(zip_df)
            if count >= 120:
                useable_zips.append(i)
        zipc = random.choice(useable_zips)
        # zipc = '93201'
        return zipc

    def prep_univariate(self, zip):
        # Read electricity data
        # Filter on Residential
        # Filter on Zipcode
        # Generate Date from month and year and keep 2 columns date and totalkWh
        print('Got my zip as', zip)
        elec_df = pd.read_csv(electricity_processed_path)
        elec_df = elec_df[elec_df['customerclass'] == 'Elec- Residential']
        elec_df = elec_df[elec_df['zipcode'] == int(zip)]
        elec_df['datetime'] = pd.to_datetime(elec_df['year'].astype(str) + '-' + elec_df['month'].astype(str), format='%Y-%m')
        prepped_df = elec_df[['datetime', 'totalkWh']]
        print(prepped_df.head())
        print(prepped_df.count())
        return prepped_df

    def prep_bivariate(self, zip):
        # Read electricity data
        # Filter on Residential
        # Filter on Zipcode
        # Generate Date from month and year and keep 3 columns date, totalkWh and totalCustomers
        print('Got my zip as', zip)
        elec_df = pd.read_csv(electricity_processed_path)
        elec_df = elec_df[elec_df['customerclass'] == 'Elec- Residential']
        elec_df = elec_df[elec_df['zipcode'] == int(zipcode)]
        elec_df['datetime'] = pd.to_datetime(elec_df['year'].astype(str) + '-' + elec_df['month'].astype(str), format='%Y-%m')
        prepped_df = elec_df[['datetime', 'totalkWh', 'totalcustomers']]
        print(prepped_df.head())
        print(prepped_df.count())
        return prepped_df

    def prep_exogenous(self, zip):
        # Read electricity data, electricity price data
        # Filter on Residential on electricity data
        # Filter on Zipcode on electricity data
        # Generate Date from month and year
        # Add weekend, weekdays, and seasons
        # Keep columns Date, totalkWh, totalCustomers, weekends, weekdays, season, and electricity price

        elec_df = pd.read_csv(electricity_processed_path)
        elec_df = elec_df[elec_df['customerclass'] == 'Elec- Residential']
        elec_df = elec_df[elec_df['zipcode'] == int(zipcode)]

        # Combined with price
        elec_price_df = pd.read_csv(electricity_price_path)
        merged_df = pd.merge(elec_df, elec_price_df, left_on='year', right_on='Year', how='left')

        # adding weekend and weekday
        merged_df = merged_df[['zipcode', 'year', 'month', 'totalkWh', 'totalcustomers', 'Avg Residential Rate ($/kWh)']]
        merged_df = merged_df.rename(columns={"Avg Residential Rate ($/kWh)": "electricity_price"})
        merged_df['weekdays'] = merged_df.apply(lambda row: self.get_weekdays(row['year'], row['month']), axis=1)
        merged_df['weekends'] = merged_df.apply(lambda row: self.get_weekends(row['year'], row['month']), axis=1)

        # adding season
        # Define dictionary to map month number to season
        season_dict = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                       3: 'Spring', 4: 'Spring', 5: 'Spring',
                       6: 'Summer', 7: 'Summer', 8: 'Summer',
                       9: 'Fall', 10: 'Fall', 11: 'Fall'}

        merged_df['season'] = merged_df['month'].map(season_dict)
        print(merged_df.head())
        merged_df['datetime'] = pd.to_datetime(
            merged_df['year'].astype(str) + '-' + merged_df['month'].astype(str), format='%Y-%m')
        prepped_df = merged_df[['datetime', 'totalkWh', 'totalcustomers', 'electricity_price', 'weekdays', 'weekends', 'season']]

        print('Got my zip as', zip)
        print(prepped_df.head(), prepped_df.count())
        return prepped_df

    def prepare_with_type(self, zipcode):
        if self.type == 'univariate':
            df = self.prep_univariate(zipcode)
        elif self.type == 'bivariate':
            df = self.prep_bivariate(zipcode)
        elif self.type == 'exogenous':
            df = self.prep_exogenous(zipcode)
        return df


cluster_for_zipcode = 1
cluster_filename = 'zipcodes-cluster-'+str(cluster_for_zipcode)+'.txt'
typ = 'bivariate'
electricity_processed_path = 'datasets/processed/electricity_combined.csv'
electricity_price_path = 'datasets/PG&E Rate Hikes vs Inflation.csv'
data_prepared_path = 'datasets/prepared/'
cluster_filepath = 'datasets/clusters/'+cluster_filename

data_prep_obj = DataPreparation(typ)
zipcode = data_prep_obj.pick_zipcode_from_cluster(cluster_filepath)
write_df = data_prep_obj.prepare_with_type(zipcode)

write_df.to_csv(data_prepared_path+typ+'_cluster_'+str(cluster_for_zipcode)+'.csv', index=False)



