# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 13:28:04 2023

@author: 35385
"""
# import libraries
import pandas as pd
import csv
import numpy as np

#number of blocks
block_number=111


#list for each household data that includes dataframe for each individual
households=[]

#weather obsolute path address
weahther_path=r'/var/share/rs1/LCL_DATA/raw_LCL_DATA/weather_hourly_darksky.csv'

#read the weather data 
weather=pd.read_csv(weahther_path)

#print the description of the weather csv file 
print('\t \t  \t***the weather file summery*** \n',weather.describe())

#Convert the time column in weather file to date format
weather['time']=pd.to_datetime(weather['time'])
weather['year'] = weather['time'].dt.year
weather['month'] = weather['time'].dt.month
weather['day'] = weather['time'].dt.day
weather['hour']=weather['time'].dt.hour

# holydays file path
holydays_path=r'/var/share/rs1/LCL_DATA/raw_LCL_DATA/uk_bank_holidays.csv'

#read the holydays file
holydays=pd.read_csv(holydays_path)
#convert the time column to date format
holydays['Bank holidays']=pd.to_datetime(holydays['Bank holidays']).dt.date
holydays=holydays['Bank holidays']

#read the acron data 
acron=pd.read_csv(r'/var/share/rs1/LCL_DATA/raw_LCL_DATA/informations_households.csv')


#print the description of the holydays csv file 
print('\t \t  \t***the weather file summery*** \n',holydays.describe())

#convert string tpye columns to float and clean the data
def to_numeric_(dataframe, columns_list):
    for column in columns_list:
        dataframe[column]= pd.to_numeric(dataframe[column], errors='coerce')
        #check if there is any null value and get a count of null values
        print(f'\n number of null values in the{column} column is',dataframe[column].isnull().sum())
        # Linear interpolation to fill null values
        dataframe[column] = dataframe[column].interpolate()
        
#coputing the mean and median of a columns of the data frome
def mean_median_colums(data, columns):
    for column in columns:
        print('mean',data[column].mean(),'******','median',data[column].median())
        
#create sperate dataframe for each hosuehold
def house_holds(dataframe,block):
    #get the IDs of households in the dataframe
    households_uniques=dataframe['LCLid'].unique()   
    #print the number of unique households in a dataframe
    print(f'number of unique households in the block_{block} is ', len(households_uniques))    
    #dreate a list of dataframes for each houshold
    for house in households_uniques:
        househod_dataframe=dataframe[dataframe['LCLid']==house]
        households.append(househod_dataframe)
        # Save the DataFrame as a CSV file
        househod_dataframe.to_csv(fr'/var/share/rs1/LCL_DATA/preparded_houshold_data/block{block}_{house}.csv')
    for house in households:
        print(f'number of dataponts for indivudual household{house} is ',len(house['LCLid']))
        
        
#list of columns to be converted to numeric and get clean
to_numeric_list=['energy(kWh/hh)', 'visibility','windBearing','temperature','dewPoint','pressure',
                 'apparentTemperature','windSpeed','humidity']
   
''' Create a dataset for a block'''


for block in range(0,block_number):
    #read the csv file and print describtion
    df0=pd.read_csv( fr'/var/share/rs1/LCL_DATA/raw_LCL_DATA/halfhourly_dataset/halfhourly_dataset/block_{block}.csv')

    #print describtion
    print(f'\t \t  \t***the block{block} file summery*** \n',df0.describe())
     
    # convert the time column to date format
    df0['tstp']=pd.to_datetime(df0['tstp'])

    # Extract year, month, and hour
    df0['year'] = df0['tstp'].dt.year
    df0['month'] = df0['tstp'].dt.month
    df0['day'] = df0['tstp'].dt.day
    df0['hour'] = df0['tstp'].dt.hour

    #merge two dataframes based on their data with hour precision
    weather_merged = pd.merge(df0, weather, on=['year', 'month', 'day','hour'], how='left')

    #create a mask for holydays
    mask=weather_merged['tstp'].dt.date.isin(holydays)


    #add holydays to the dataframe, 1 for holydays and zero for working days
    weather_merged['holydays']=mask.astype(int)
    
    weather_merged['weekday']=weather_merged['tstp'].dt.weekday

    #update the holydays column for weeknds 
    weather_merged.loc[weather_merged['weekday']>=5,'holydays']=1

    #merge two dataframes  based on their household ID
    weather_merged = pd.merge(weather_merged, acron, on=['LCLid'], how='left')

    print(f'number of the rows in entire data dataframe for block{block}',len(weather_merged['LCLid']))

    #convert and clean columns
    to_numeric_(weather_merged,to_numeric_list)
    #create seperate dataframes for each house hold
    house_holds(weather_merged,block)
    


