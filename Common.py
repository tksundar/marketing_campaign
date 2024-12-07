import datetime
import warnings

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy.stats import normaltest

pd.options.mode.copy_on_write = True
from sklearn.preprocessing import  OrdinalEncoder,LabelEncoder

from prettytable import prettytable, PrettyTable

DONE = 'Done\n'
def drop_outlier_age_records(data):
    '''
    Drop records with outlier age values
    :param data:
    :return:
    '''
    if is_normal(data['Age']):
        outliers = detect_outliers_z(data['Age'])
    else:
        outliers = detect_outliers_iqr(data['Age'])
    print('\tDropping records for the following age outliers\n\t',outliers)
    index_names = data[(data['Age'] >= min(outliers))].index
    data = data.drop(index_names)
    return data

def is_normal(data):
    """
    checks if the data is normally distributed
    """

    alpha = 0.005
    _, p = normaltest(data)
    if p > alpha:
        return True
    return False

def label_encode(df:DataFrame):
    '''

    :param df: a dataframe
    :return:
    '''
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(df)

def ordinal_encode(df:DataFrame):
    '''
    :param df: Dataframe
    :return:
    '''
    encoder = OrdinalEncoder()
    return encoder.fit_transform(df)



def get_income(row, data):
    key = (row['Marital_Status'], row['Education'])
    return data.get(key)


def get_marital_status(row,mode):
    if row['Marital_Status'] == 'Alone' or row['Marital_Status'] == 'YOLO' or row['Marital_Status'] == 'Absurd':
        return mode
    else:
        return row['Marital_Status']


def print_test_header(H0, H1):
    print('*************************************************************************************')
    print('H0', H0)
    print('H1', H1)
    print('*************************************************************************************')


def detect_outliers_z(data, threshold=3.0):
    """
    Detects outliers based on z score where z score is (x-mean)/std abd using threshold z value as 3
    :param data:
    :return: list of outliers
    """
    mean = np.mean(data)
    std = np.std(data)
    outliers = []
    for i in data:
        z_score = (i - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers


def detect_outliers_iqr(data, factor=1.5):
    """
    Using the 1.5 * IQR range to detect outliers. IQR is preferred to Z score
    where distribution is not normal and we use median imputation
    :param data:
    :return: outliers array
    """
    data = sorted(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    IQR = q3 - q1
    lwr_bound = q1 - (factor * IQR)
    upr_bound = q3 + (factor * IQR)
    outliers = []
    for i in data:
        if i < lwr_bound or i > upr_bound:
            outliers.append(i)
    return outliers


def remove_outliers(data, outlier_columns, threshold=3, factor=1.5):
    """
    :param factor: factor for outlier removal with IQR
    :param threshold: threshold for z score based outlier removal
    :param data:
    :param outlier_columns:
    :return: data with outliers replaced
    """
    for column in outlier_columns:
        array = data[column]
        if not is_normal(array):
            outliers = detect_outliers_iqr(array, factor)
            value = np.median(array)
        else:
            outliers = detect_outliers_z(array.data, threshold)
            value = np.mean(array)
        for i in outliers:
            # for every i(outlier) in array , replace it by value else retain the same array element
            array = np.where(array == i, value, array)
            data[column] = np.asarray(array)
    return data


def get_age_category(row):
    if row['Age'] < 30: return 'below 30'
    elif 30 < row['Age'] < 60: return '30 - 60'
    else: return 'Above 60'



def get_table(dfs):
        table = PrettyTable()
        columns =  dfs.columns.tolist()
        table.field_names =columns
        for index,row in dfs.iterrows():
            values = []
            for column in columns:
              values.append(row[column])
            table.add_row(values)

        return table

def clean_data(mkt_data):
    """
     Cleans the input by doing the following
     1. Strips all whitespace
     2. replaces nan values with appropriate values
     3. Converts all date columns to pandas datetime dtype
     4. Cleans Marital_Status column by replacing bad values with sensible ones
     5. Cleans the Income column by
        a)Removing $ sign from the entries and converting the dtype to float64 so that mathematical operations are possible
        b)Replacing nan entries(24 of them) with suitable mean values of income based on marital status and education level of the customer
     6. Removed outlier records for age
    :param mkt_data: DataFrame
    :return:  mkt_data: DataFrane
    """

    warnings.simplefilter(action='ignore', category=FutureWarning)
    print('data shape before cleaning : ', mkt_data.shape)
    mkt_data.drop_duplicates(keep='first', inplace=True)
    print('stripping white spaces ...')

    for col in mkt_data.columns:
        if ' ' in col:
            mkt_data.rename(columns={col: col.strip()}, inplace=True)
    for col in mkt_data.columns:
        if is_object_dtype(mkt_data[col]):
            mkt_data[col] = mkt_data[col].str.strip()
    print(DONE)

    print('Fixing income column...')

    mkt_data['Income'] = mkt_data['Income'].str.replace('$', '').str.replace(',', '', regex=True).astype(float)
    income_by_ms_and_eduction = pd.DataFrame(mkt_data.groupby(['Marital_Status', 'Education']).Income.mean())
    # incomes is a dictionary where key is a tuple and value is the mean for that combination like -> ('Married', 'Master'): 53286.02898550725
    # We will use this dictionary to replace nan income cells with appropriate values. This is based on the problem statement
    # Another way is to use country and education to replace the income values
    incomes = income_by_ms_and_eduction.to_dict().get('Income')
    # We fill the missing values in a row by the mean value of income for that row's marital_status and eduction level.
    # we use the incomes dictionary for lookup
    mkt_data.loc[:, 'Income'] = mkt_data.apply(
        lambda row: get_income(row, incomes) if np.isnan(row.loc['Income']) else row.loc['Income'], axis=1)
    print(DONE)

    print('Fixing the dateformat...')

    mkt_data['Dt_Customer'] = pd.to_datetime(mkt_data['Dt_Customer'], format="%m/%d/%y")
    print(DONE)
    #Fixing nonsensical marital status values with the mode of the column
    print('Fixing marital_status column')

    marital_status_mode = pd.DataFrame(mkt_data['Marital_Status'].mode()).iloc[0]['Marital_Status']
    mkt_data['Marital_Status'] = mkt_data.apply(lambda row: get_marital_status(row,marital_status_mode) , axis=1)
    print(DONE)


    total_across_products = mkt_data.apply(
        lambda row: row['MntWines'] + row['MntFruits'] + row['MntMeatProducts'] + row['MntFishProducts'] + row[
            'MntGoldProds'], axis=1)
    total_across_channels = mkt_data.apply(
        lambda row: row['NumDealsPurchases'] + row['NumWebPurchases'] + row['NumCatalogPurchases'] + row[
            'NumStorePurchases'] + row['MntGoldProds'], axis=1)
    # We use the ordinal encoder because education is an ordinal category where educational qualifications
    # have an order with 2nd cycle being the lowest and Ph.D being the highest
    print('Encoding Education')
    encoded = ordinal_encode(pd.DataFrame(mkt_data['Education']))
    print(DONE)
    print('Encoding country')
    mkt_data['ECountry'] = label_encode(mkt_data['Country'])
    print(DONE)
    print('Adding columns...')
    # adding first column
    mkt_data.loc[:, 'TotalAcrossProducts'] = total_across_products
    print('\tAdded column TotalAcrossProducts')
    #Adding 2nd column
    mkt_data.loc[:, 'TotalAcrossChannels'] = total_across_channels
    print('\tAdded column TotalAcrossChannels')
    off_store_purchases = mkt_data.apply(lambda row: row['TotalAcrossChannels'] - row['NumStorePurchases'], axis=1)
    #adding 3rd column
    mkt_data.loc[:, 'OffStorePurchases'] = off_store_purchases
    print('\tAdded column OffStorePurchases')
    #adding 4th column
    #mkt_data.loc[:, 'ECountry'] = e_ctr
    print('\tAdded column ECountry')

     #adding 5th column
    mkt_data.loc[:, 'EEducation'] = encoded
    print('\tAdded column EEducation')
    current_year = datetime.datetime.now().year
    #adding 6th column
    mkt_data.loc[:, 'Age'] = mkt_data.apply(lambda row: current_year - row['Year_Birth'], axis=1)
    print('\tAdded column Age')
    age_category =mkt_data.apply(lambda row : get_age_category(row),axis=1)
    #Adding 7th Column
    mkt_data.loc[:,'AgeBracket'] = age_category
    print('\tAdded column AgeBracket')
    print(DONE)
    # drop outlier age records
    print('Dropping outlier age records')
    mkt_data = drop_outlier_age_records(mkt_data)
    print(DONE)
    # check that there are no more missing values
    missing = mkt_data.isna()
    if True in missing.values:
        raise ValueError('Missing values present')
    print(f'data shape after cleaning {mkt_data.shape}')


    return mkt_data
def get_channels_and_products():
    copy = dep_vars.copy()
    copy.extend(products)
    return copy

threshold = 3.0  # for outlier removal using z score for normal distributions
factor = 1.5  # for outlier removal using IQR interval for skewed distributions
products = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
prod_labels = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
dep_vars = ['TotalAcrossProducts', 'TotalAcrossChannels', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
            'NumStorePurchases']

ind_vars = ['AgeBracket', 'Education', 'Kidhome','Country']
hues = ['Education','Kidhome','Country','AgeBracket']
ind_vars_1 = ['Age', 'Education', 'Kidhome','Country']


