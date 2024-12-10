"""
Filename: marketing_campaign.ipynb
Author: Sundar Krishnamachari
Date: 2024-12-14
Version: 1.0
Description: This is the source code for the project fof PGC AIML: Applied Data Science with Python course
"""
import copy
import datetime

import sys

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.dtypes.common import is_object_dtype
from prettytable import PrettyTable
from scipy.stats import pearsonr, normaltest
from statsmodels.stats.weightstats import ztest
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

pd.options.mode.copy_on_write = True
matplotlib.pyplot.set_loglevel(level='warning')
#### Global read only variables

threshold = 3.0  # for outlier removal using z score for normal distributions
factor = 1.5  # for outlier removal using IQR interval for skewed distributions
products = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
prod_labels = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
dep_vars = ['TotalAcrossProducts', 'TotalAcrossChannels', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
            'NumStorePurchases']
ind_vars = ['AgeBracket', 'Education', 'Kidhome', 'Country']
ind_vars_1 = ['Age', 'Education', 'Kidhome', 'Country']
hues = ['Education', 'Kidhome', 'Country', 'AgeBracket']

low = '< 25K'
lm = '25K - 45K'
middle = '45K - 65K'
high = ' > 65K'

menu = ['Analyse Data', 'Do hypothesis Tests', 'Do Correlations', 'Perform All Tasks']
DONE = 'Done\n'
# field_names = ['Task', 'Analysis', 'Tests', 'Visualisations', 'Other Correlations', 'Grouped charts', 'All']
# options = ['Choice', 1, 2, 3, 4, 5, 6]
field_names = ['Task ➤', 'Analysis', 'Tests', 'Visualisations', 'Other Correlations', 'Grouped charts', 'All']
options = ['Choice ➤', 1, 2, 3, 4, 5, 6]

def plot_hist(data,column,ax,title=None):
    plot = sns.histplot(data, x=column, kde=True, ax=ax)
    xticks = plot.get_xticklabels()
    plot.set_xticks(plot.get_xticks())
    plot.set_xticklabels(xticks, rotation=90)
    plot.set_title(title)

def plot_box(data,column,ax,title=None):
    plot = sns.boxplot(data, y=column,  ax=ax)
    xticks = plot.get_xticklabels()
    plot.set_xticks(plot.get_xticks())
    plot.set_xticklabels(xticks, rotation=90)
    plot.set_title(title)

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
    print('\tDropping records for the following age outliers\n\t', outliers)
    index_names = data[(data['Age'] >= min(outliers))].index
    data = data.drop(index_names)
    return data


def get_title(df, column1, column2):
    tokens = []
    for index, row in df.iterrows():
        tokens.append(str(int(row[column1])))
        tokens.append(' = ')
        tokens.append(row[column2])
        tokens.append('     ')
    return ''.join(tokens)


def is_normal(data):
    """
    checks if the data is normally distributed
    """

    alpha = 0.005
    _, p = normaltest(data)
    if p > alpha:
        return True
    return False


def label_encode(df):
    '''

    :param df: a dataframe
    :return:
    '''
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(df)


def ordinal_encode(df):
    '''
    :param df: Dataframe
    :return:
    '''
    encoder = OrdinalEncoder()
    return encoder.fit_transform(df)


def get_income(row, data):
    key = (row['Marital_Status'], row['Education'])
    return data.get(key)


def get_marital_status(row, mode):
    if row['Marital_Status'] == 'Alone' or row['Marital_Status'] == 'YOLO' or row['Marital_Status'] == 'Absurd':
        return mode
    else:
        return row['Marital_Status']


def print_test_header(H0, H1):
    print('*************************************************************************************')
    print('H0:', H0)
    print('H1:', H1)
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
    if row['Age'] < 30:
        return '< 30'
    elif 30 < row['Age'] < 60:
        return '30 - 60'
    else:
        return '> 60'


def get_table(dfs):
    table = PrettyTable()
    columns = dfs.columns.tolist()
    table.field_names = columns
    for index, row in dfs.iterrows():
        values = []
        for column in columns:
            values.append(row[column])
        table.add_row(values)

    return table


def color_max_min(ax, values):
    for bar in ax.patches:
        if bar.get_height() == max(values):
            bar.set_color('skyblue')
        elif bar.get_height() == min(values):
            bar.set_color('grey')
        else:
            bar.set_color('orange')


def get_income_bracket(row):
    income = row['Income']
    if income < 25000: return low
    if 25000 < income < 45000:  return lm
    if 45000 < income < 65000:  return middle
    if income > 65000:  return high


def get_encoded_income_range(row):
    if row['IncomeBracket'] == low: return 1
    if row['IncomeBracket'] == lm: return 2
    if row['IncomeBracket'] == middle: return 3
    if row['IncomeBracket'] == high: return 4


def get_encoded_age_bracket(row):
    if row['AgeBracket'] == '< 30': return 0
    if row['AgeBracket'] == '30 - 60': return 1
    if row['AgeBracket'] == '> 60': return 2


def plot_correlations_for(data, dep):
    ind_vars__ = ['EEducation' if element == 'Education' else element for element in ind_vars_1]
    ind_vars__ = ['ECountry' if element == 'Country' else element for element in ind_vars__]
    ind_vars__ = ['EIncomeBracket' if element == 'IncomeBracket' else element for element in ind_vars__]

    edn = pd.DataFrame(data.groupby(['EEducation', 'Education']).TotalAcrossProducts.sum()).reset_index()
    e_title = get_title(edn, 'EEducation', 'Education')
    ctr = pd.DataFrame(data.groupby(['ECountry', 'Country']).TotalAcrossProducts.sum()).reset_index()
    c_title = get_title(ctr, 'ECountry', 'Country')
    inc = pd.DataFrame(data.groupby(['EIncomeBracket', 'IncomeBracket']).TotalAcrossProducts.sum()).reset_index()
    i_title = get_title(inc, 'EIncomeBracket', 'IncomeBracket')

    for i, x in enumerate(ind_vars__):
        fig, axes = plt.subplots(nrows=1, ncols=len(dep), figsize=(16, 6), layout='constrained')
        if x == 'EEducation':
            plt.suptitle(e_title)
        elif x == 'ECountry':
            plt.suptitle(c_title)
        elif x == 'EIncomeBracket':
            plt.suptitle(i_title)
        for j, y in enumerate(dep):
            df = pd.DataFrame(data.groupby(x)[y].sum()).reset_index()
            cor, _ = pearsonr(df[x], df[y])
            sns.regplot(x=x, y=y, data=df, line_kws={"color": get_color(cor)}, ax=axes[j]).set_title(
                'corr = %.4f' % cor)

    plt.show()


def correlation_of_spending_patterns(data):
    print('*************************************************************************************')
    print('SCorrelations between various variables')
    plot_correlations_for(data, dep_vars)
    plot_correlations_for(data, products)


def grouped_bar_chart(data):
    print('****************************************************************************')
    print('Additional Grouped charts')
    hues = ['Education', 'Kidhome', 'Country', 'AgeBracket', 'IncomeBracket']
    len1 = int(len(dep_vars) / 2)
    len2 = int(len(products) / 2)
    columns1 = dep_vars[0:len1]
    columns2 = dep_vars[len1:len1 * 2]
    ana = Analysis(data)
    ana.plot_bar_plot(ind_vars, columns1, hues)
    ana.plot_bar_plot(ind_vars, columns2, hues)
    prd1 = products[0:len2]
    prd2 = products[len2:len2 * 2]
    ana.plot_bar_plot(ind_vars, prd1, hues)
    ana.plot_bar_plot(ind_vars, prd2, hues)


def cleanup_income_column(mkt_data, regex=False):
    return mkt_data['Income'].str.replace('$', '', regex=regex).str.replace(',', '', regex=regex).astype(float)


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

    mkt_data['Income'] = cleanup_income_column(mkt_data)
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

    # Fixing nonsensical marital status values with the mode of the column
    print('Fixing marital_status column')
    marital_status_mode = pd.DataFrame(mkt_data['Marital_Status'].mode()).iloc[0]['Marital_Status']
    mkt_data['Marital_Status'] = mkt_data.apply(lambda row: get_marital_status(row, marital_status_mode), axis=1)
    print(DONE)

    income_brackets = mkt_data.apply(lambda row: get_income_bracket(row), axis=1)

    print('Encoding Education')
    e_education = ordinal_encode(pd.DataFrame(mkt_data['Education']))
    print(DONE)
    print('Encoding country')
    e_ctr = label_encode(mkt_data['Country'])
    print(DONE)
    print('Encoding Marital_Status')
    e_marital_status = label_encode(mkt_data['Marital_Status'])
    print(DONE)

    print('Adding columns...')

    total_across_products = mkt_data.apply(
        lambda row: row['MntWines'] + row['MntFruits'] + row['MntMeatProducts'] + row['MntFishProducts'] + row[
            'MntGoldProds'], axis=1)
    total_across_channels = mkt_data.apply(
        lambda row: row['NumDealsPurchases'] + row['NumWebPurchases'] + row['NumCatalogPurchases'] + row[
            'NumStorePurchases'] + row['MntGoldProds'], axis=1)
    # We use the ordinal encoder because education is an ordinal category where educational qualifications
    # have an order with 2nd cycle being the lowest and Ph.D being the highest

    # adding first column
    mkt_data.loc[:, 'TotalAcrossProducts'] = total_across_products
    print('\t1.Added column TotalAcrossProducts')

    # Adding 2nd column
    mkt_data.loc[:, 'TotalAcrossChannels'] = total_across_channels
    print('\t2.Added column TotalAcrossChannels')
    off_store_purchases = mkt_data.apply(lambda row: row['TotalAcrossChannels'] - row['NumStorePurchases'], axis=1)

    # adding 3rd column
    mkt_data.loc[:, 'OffStorePurchases'] = off_store_purchases
    print('\t3.Added column OffStorePurchases')

    # adding 4th column
    mkt_data.loc[:, 'ECountry'] = e_ctr
    print('\t4.Added column ECountry')

    # adding 5th column
    mkt_data.loc[:, 'EEducation'] = e_education
    print('\t5.Added column EEducation')

    # 6
    mkt_data.loc[:, 'EMarital_Status'] = e_marital_status
    print('\t6.Added column EMarital_Status')
    current_year = datetime.datetime.now().year

    # 7
    mkt_data.loc[:, 'Age'] = mkt_data.apply(lambda row: current_year - row['Year_Birth'], axis=1)
    print('\t7.Added column Age')

    # 8
    age_category = mkt_data.apply(lambda row: get_age_category(row), axis=1)
    mkt_data.loc[:, 'AgeBracket'] = age_category
    print('\t8.Added column AgeBracket')

    # 9
    mkt_data.loc[:, 'EAgeBracket'] = mkt_data.apply(lambda row: get_encoded_age_bracket(row), axis=1)
    print('\t9.Added column EAgeBracket')

    # 10
    mkt_data.loc[:, 'IncomeBracket'] = income_brackets
    print('\t10.Added Column IncomeBracket')

    # 11
    mkt_data.loc[:, 'EIncomeBracket'] = mkt_data.apply(lambda row: get_encoded_income_range(row), axis=1)
    print('\t11.Added column EIncomeBracket ')

    # 12
    mkt_data.loc[:, 'TotalChildren'] = mkt_data.apply(lambda row: row["Kidhome"] + row['Teenhome'], axis=1)
    print('\t12.Added column TotalChildren')
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


class Analysis:

    def __init__(self, data):
        self.data = data

    def key_insights(self):
        """
        Analyzes the data and displays key insights such as
            1. Income vs expenditure across products and sales channels and determine correlations
            2. Education vs expenditure across products and sales channels and determine correlations
            3. Marital Status vs expenditure across products and sales channels and determine correlations
            4. Age vs expenditure across products and sales channels and determine correlations
        :return:
        """
        print('****************************************')
        print('Key insights')

        variables = ['IncomeBracket', 'Education', 'Marital_Status', 'AgeBracket']
        values = copy.deepcopy(dep_vars)
        values.extend(products)
        self.plot_bar_plot(variables, values)
        self.plot_box_and_histogram()
        for_heat_map = ['EIncomeBracket', 'EEducation', 'EMarital_Status', 'Age']
        title = 'Correlation with total expenditure and sales channels'
        self.plot_hm(for_heat_map, dep_vars, title=title)
        title = 'Correlation with product lines'
        self.plot_hm(for_heat_map, products, title=title)

    def plot_bar_plot(self, x_values, y_values, hues=None):
        print('****************************************')
        print('High level relationship between key variables')
        for i, x in enumerate(x_values):
            fig, axes = plt.subplots(nrows=1, ncols=len(y_values), figsize=(16,12), layout='constrained')
            for j, y in enumerate(y_values):
                if hues is not None:
                    title = ' %s grouped by %s ' % (x, hues[i])
                    df = pd.DataFrame(self.data.groupby([x, hues[i]])[y].mean()).reset_index()
                    bars = sns.barplot(df, x=x, y=y, hue=hues[i], ax=axes[j])
                    bars.legend(loc='upper left', bbox_to_anchor=(1, 1))
                else:
                    df = pd.DataFrame(self.data.groupby(x)[y].mean()).reset_index()
                    title = ' %s vs %s' % (x, y)
                    bars = sns.barplot(df, x=x, y=y, ax=axes[j])
                    color_max_min(axes[j], df[y].tolist())
                xticks = bars.get_xticklabels()
                bars.set_xticks(bars.get_xticks())
                bars.set_xticklabels(xticks, rotation=90)
            # 0bars.set_title(title)

            plt.show()

    def plot_box_and_histogram(self):
        print('****************************************************************************')
        print('distribution charts before and after outlier treatment')

        values = copy.deepcopy(dep_vars)
        values.extend(products)
        before = 'Before outlier treatment'
        after = 'After outlier treatment'
        for column in values:
            fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 10), layout='constrained')
            plot_hist(self.data,column,ax[0],before)
            plot_box(self.data,column,ax[1],before)
            data = copy.deepcopy(self.data)
            remove_outliers(data, [column])
            plot_hist(data, column, ax[2], after)
            plot_box(data, column, ax[3], after)
            plt.suptitle('%s Distribution is positively skewed\nIQR range and median values used for replacement'%column)
            plt.show()

    def plot_hm(self, variables, values, title=None):
        print('****************************************')
        print('Heatmap showing %s' % title)
        cmap = 'Paired'
        data = remove_outliers(self.data, values)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 16), layout='constrained')
        v_copy = copy.deepcopy(variables)
        v_copy.extend(values)
        df = data[v_copy]
        df.rename(columns={"EIncomeBracket": 'IncomeBracket',
                           "EEducation": "Education",
                           "EMarital_Status": "Marital_Status"
                           }, inplace=True)

        heatmap = sns.heatmap(df.corr(), annot=True, cmap=cmap, ax=axes)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
        heatmap.set_title(title)
        plt.show()


class Tests:

    def __init__(self, data):
        self.data = data

    def test_older_individuals_rely_on_store_shopping(self):
        """
       Here we use z test to confirm or reject the null hypothesis.
        H0 = 'There is no difference in store shopping pattern between older and younger individuals'
        H1 = 'Older individuals rely on store shopping compared to younger individuals'
       """
        data = self.data
        H0 = 'There is no difference in store shopping pattern between older and younger individuals'
        H1 = 'Older individuals rely on store shopping compared to younger individuals'
        print_test_header(H0, H1)
        alpha = 0.001
        age_threshold = 60
        old_purchases = data[data['Age'] >= age_threshold].loc[:, ['Age', 'NumStorePurchases']]
        young_purchases = data[data['Age'] < age_threshold].loc[:, ['Age', 'NumStorePurchases']]
        _, p = ztest(young_purchases['NumStorePurchases'], old_purchases['NumStorePurchases'])

        pop_mean = data['NumStorePurchases'].mean()
        young_mean = young_purchases['NumStorePurchases'].mean()
        old_mean = old_purchases['NumStorePurchases'].mean()
        if p > alpha:
            title = H0
        else:
            title = H1

        f, a = plt.subplots(1, 1, figsize=(10, 6))
        dic = {'Old Store Purchases': [old_mean], 'Young Store Purchases': [young_mean]}
        df = pd.DataFrame(dic)
        sns.barplot(data=df, ax=a)
        plt.suptitle(
            '%s.\np value = %.10f,\n alpha = %.4f \n population mean = %.4f , \nyoung_mean = %.4f,\n old_mean = %.4f\n'

            % (title, p, alpha, pop_mean, young_mean, old_mean))
        plt.tight_layout()
        plt.show()

    def plot_purchase_means(self, labels, H0, H1, pop_mean, data_with_kids, data_without_kids, ax):
        alpha = 0.001
        _, p = ztest(data_without_kids, data_with_kids)
        if p > alpha:
            title = H0
        else:
            title = H1
        with_kids_mean = data_with_kids.mean()
        without_kids_mean = data_without_kids.mean()
        values = [with_kids_mean, without_kids_mean]
        sns.barplot(y=values, x=labels, ax=ax)
        t = (
                '%s\np value = %.5f\nalpha = %.4f, \nPopulation mean=%.4f,\n with_kids_mean = %.4f,\nwithout_kids_mean=%.4f\n'
                % (title, p, alpha, pop_mean, with_kids_mean, without_kids_mean))
        ax.set_title(t)
        plt.tight_layout()

    def test_kids_influence_online_shopping(self):
        """
        Does a Z test on the sample with kids(kids+teens) and the sample with no kids
        """
        data = self.data
        H0 = 'Customers with kids prefer  web purchases'
        H1 = 'Customers with kids do not prefer web  purchases'

        print_test_header(H0, H1)
        num_web_purchases_with_kids = data[data['Kidhome'] > 0].loc[:, ['Kidhome', 'NumWebPurchases']]
        num_web_purchases_without_kids = data[data['Kidhome'] == 0].loc[:, ['Kidhome', 'NumWebPurchases']]
        pop_mean = data['NumWebPurchases'].mean()
        data_with_kids = num_web_purchases_with_kids['NumWebPurchases']
        data_without_kids = num_web_purchases_without_kids['NumWebPurchases']
        labels = ['With Kids', 'without kids']

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        self.plot_purchase_means(labels, H0, H1, pop_mean,
                                 data_with_kids,
                                 data_without_kids,
                                 axes[0])
        axes[0].set_ylabel('Web Purchases')

        H0 = 'Customers with kids prefer  off store purchases'
        H1 = 'FTR - Customers with kids do not prefer off store purchases'
        # off_store_purchases = data.apply(lambda row: row['TotalAcrossChannels'] - row['NumStorePurchases'], axis=1)
        # data.loc[:, 'OffStorePurchases'] = off_store_purchases
        off_store_with_kids = data[data['Kidhome'] > 0].loc[:, ['Kidhome', 'OffStorePurchases']]
        off_store_without_kids = data[data['Kidhome'] == 0].loc[:, ['Kidhome', 'OffStorePurchases']]
        pop_mean = data['OffStorePurchases'].mean()
        labels = ['With Kids', 'without kids']
        self.plot_purchase_means(labels, H0, H1, pop_mean,
                                 off_store_with_kids['OffStorePurchases'],
                                 off_store_without_kids['OffStorePurchases'],
                                 axes[1])
        axes[1].set_ylabel('Off Store Purchases')
        axes[1].yaxis.set_label_position("right")
        axes[1].yaxis.tick_right()

        plt.show()

    def test_cannibalization_by_off_store_sales_channels(self):
        """
        Assumption : All channels other than store purchases( such as web, deals or catalogue purchases) are treated as off store sales
        This may not be true as discount(deal) and catalogue sales can be from the store too.
        """
        data = self.data
        H0 = 'Off Store channels do not cannibalize store purchases'
        H1 = 'Off store channels cannibalize store purchases'
        print_test_header(H0, H1)
        alpha = 0.001
        # total_off_store_purchases = data.apply(lambda row: row['TotalAcrossChannels'] - row['NumStorePurchases'],
        #                                       axis=1)
        # data.loc[:, 'OffStorePurchases'] = total_off_store_purchases
        _, p = ztest(data['OffStorePurchases'], data['NumStorePurchases'])
        if p > alpha:
            title = H0
        else:
            title = H1
        fig, a = plt.subplots(1, 1, figsize=(10, 6))
        store_purchases_mean = data['NumStorePurchases'].mean()
        off_store_purchases_mean = data['OffStorePurchases'].mean()
        values = [store_purchases_mean, off_store_purchases_mean]
        labels = ['Store Purchases', 'Off Store Purchases']
        sns.barplot(x=labels, y=values, ax=a)
        plt.suptitle(
            "%s\n store_purchases_mean = %.4f , \noff_store_purchases_mean = %.4f ,\n p value = %.10f,\nalpha = %.4f"
            % (title, store_purchases_mean, off_store_purchases_mean, p, alpha))
        plt.tight_layout()
        plt.show()

    def test_us_outperforms_purchases(self):
        """
        H0 : US doesn't outperform the rest of the countries in purchase volumes
        H1 : US  outperforms the rest of the countries in purchase volumes
        :param data:
        :return:
        """
        data = self.data
        H0 = 'United States does not significantly outperform the rest of the world in total purchase volumes'
        H1 = 'United States significantly outperform the rest of the world in total purchase volumes'
        print_test_header(H0, H1)
        alpha = 0.001
        purchases_by_country = data.loc[:, ['TotalAcrossProducts', 'Country']]

        pop_mean = purchases_by_country['TotalAcrossProducts'].mean()
        us_purchases = data[data['Country'] == 'US']
        us_mean = us_purchases['TotalAcrossProducts'].mean()
        _, p = ztest(us_purchases['TotalAcrossProducts'], value=pop_mean)
        if p > alpha:
            title = H0
        else:
            title = H1
        by_country = pd.DataFrame(data.groupby('Country').TotalAcrossProducts.mean()).reset_index()
        by_country['TotalAcrossProducts'] = by_country.rename(columns={'TotalAcrossProducts': 'MeanAcrossProducts'},
                                                              inplace=True)
        fig, a = plt.subplots(1, 1, figsize=(10, 6), layout='constrained')
        plot = sns.barplot(x='Country', y='MeanAcrossProducts', hue='Country', data=by_country, ax=a)
        plot_title = '%s\nPopulation mean = %.4f,\n US mean = %.4f,\n p value = %.4f,\nalpha %.4f' % (
            title, pop_mean, us_mean, p, alpha)
        plot.set_title(plot_title)
        encoded = pd.get_dummies(data, 'Country')
        corr, _ = pearsonr(encoded['Country_US'], encoded['TotalAcrossProducts'])
        print(corr)
        plt.show()


def get_color(corr):
    green_threshold = 0.1
    red_threshold = -0.1
    if corr > green_threshold: return 'green'
    if red_threshold < corr < green_threshold: return 'blue'
    if corr < red_threshold: return 'red'


class Visualisations:

    def __init__(self, data):
        self.data = data

    def get_correlation(self, corr):
        weak_threshold = .3
        medium_threshold = .6

        if abs(corr) < weak_threshold:
            if corr < 0:
                return 'weak  %s correlation' % 'negative'
            else:
                return 'weak  %s correlation' % 'positive'

        elif weak_threshold < abs(corr) < medium_threshold:
            if corr < 0:
                return 'medium  %s correlation' % 'negative'
            else:
                return 'medium %s correlation' % 'positive'
        else:
            if corr < 0:
                return 'strong  %s correlation' % 'negative'
            else:
                return 'strong %s correlation' % 'positive'

    def top_performer(self):

        values = []
        for product in products:
            val = self.data[product].sum()
            values.append(val)
        data = {'Products': prod_labels, 'Amount': values}
        df = pd.DataFrame(data)
        top_performer = prod_labels[values.index(max(values))]
        bottom_performer = prod_labels[values.index(min(values))]
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        title = 'Top Performer product : %s\nLowest Revenue is for:  %s ' % (top_performer, bottom_performer)
        print('******************************************************************************')
        print('Visualization 1: ', title)
        print(get_table(df))
        sns.barplot(x='Products', y='Amount', hue='Products', data=df, ax=ax).set_title(title)
        color_max_min(ax, values)
        plt.xticks(rotation=45)
        plt.show()

    def correlation_customer_age_and_campaign(self):
        data = self.data
        correlation, _ = pearsonr(data['Age'], data['Response'])
        title = self.get_correlation(correlation)
        df = pd.DataFrame(data.groupby('Age').Response.sum()).reset_index()
        print('*****************************************************************************')
        print('Visualization 2: ', title)
        print(get_table(df))

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x='Age', y='Response', data=df, ax=ax).set_title(
            'Correlation between age and campaign response.\n%s.  Pearsons Correlation Coefficient = %.4f' % (
            title, correlation))
        plt.show()

    def country_with_highest_campaign_response(self):
        df = pd.DataFrame(self.data.groupby(['Country']).Response.sum()).reset_index()
        max_response = df[df['Response'] == max(df['Response'])]
        max_response_country = max_response['Country'].tolist()[0]
        max_response_value = max_response['Response'].tolist()[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        title = 'Country with max campaign response acceptance\nCountry : %s  Accepted count = %d' % (
        max_response_country, max_response_value)
        print('**************************************************************************')
        print('Visualization 3: ', title)
        print(get_table(df))

        sns.barplot(x='Country', y='Response', data=df).set_title(title)
        color_max_min(ax, df['Response'].tolist())
        plt.show()

    def correlation_num_kids_and_total_spend(self):
        dfp = pd.DataFrame(self.data.groupby('TotalChildren').TotalAcrossProducts.mean()).reset_index()
        dfp.rename(columns={'TotalAcrossProducts': 'MeanAcrossProducts'}, inplace=True)
        corr, _ = pearsonr(dfp['TotalChildren'], dfp['MeanAcrossProducts'])

        title = 'Correlation between total children and expenditure\n%s, Pearson correlation coefficient = %.4f' % (
        self.get_correlation(corr), corr)
        print('******************************************************************************')
        print('Visualization 4: ', title)
        print(get_table(dfp))

        fig, axes = plt.subplots(1, 1, figsize=(10, 6), layout='constrained')
        sns.regplot(x='TotalChildren', y='MeanAcrossProducts', data=dfp
                    , ax=axes, line_kws={"color": "red"}).set_title(title)
        plt.show()

    def correlation_between_education_and_complaints(self):
        fig, axes = plt.subplots(1, 1, figsize=(10, 6), layout='constrained')

        df = pd.DataFrame(self.data.groupby(['EEducation', 'Education']).Complain.sum()).reset_index()
        corr, _ = pearsonr(df['EEducation'], df['Complain'])
        # dfs = pd.DataFrame(self.data.pivot_table(columns = ['Education'],values = ['Complain'],aggfunc= 'sum'))
        max_row = pd.DataFrame(df[df['Complain'] == max(df['Complain'])])
        min_row = pd.DataFrame(df[df['Complain'] == min(df['Complain'])])
        qual_max = max_row.iloc[0]['Education']
        qual_min = min_row.iloc[0]['Education']
        qual_string = 'Customers with %s degree complained the most and those with %s degree complained the least ' % (
        qual_max, qual_min)
        corr_string = self.get_correlation(corr)
        title = '%s\nThere is %s between education and complaints\nPearson Correlation Coefficient = %.4f' % (
        qual_string, corr_string, corr)
        print('********************************************************************')
        print('Visualization 5:')
        print(title)
        print(get_table(df))
        sns.regplot(x='EEducation', y='Complain', data=df, ax=axes).set_title(title)
        plt.show()


def analyse_data(mkt_data):
    print('*******************************************************************************')
    print('Starting basis data analysis for preliminary insights')
    print('*******************************************************************************')
    analysis = Analysis(mkt_data)
    analysis.key_insights()


def do_hypothesis_tests(data):
    print('*******************************************************************************')
    print('Starting hypothesis testing')
    print('*******************************************************************************')
    tests = Tests(data)
    tests.test_older_individuals_rely_on_store_shopping()
    tests.test_kids_influence_online_shopping()
    tests.test_cannibalization_by_off_store_sales_channels()
    tests.test_us_outperforms_purchases()


def do_visualisations(data):
    print('*******************************************************************************')
    print('Starting Visualisations for correlations')
    print('*******************************************************************************')
    visualise = Visualisations(data)
    visualise.top_performer()
    visualise.correlation_customer_age_and_campaign()
    visualise.country_with_highest_campaign_response()
    visualise.correlation_num_kids_and_total_spend()
    visualise.correlation_between_education_and_complaints()


def perform_all_tasks(mkt_data):
    analyse_data(mkt_data)
    do_hypothesis_tests(mkt_data)
    do_visualisations(mkt_data)
    plot_correlations(mkt_data)
    plot_grouped_charts(mkt_data)


def plot_correlations(mkt_data):
    correlation_of_spending_patterns(mkt_data)


def plot_grouped_charts(mkt_data):
    grouped_bar_chart(mkt_data)

class Main:
    print('*******************************************************************************')
    print('Marketing Campaign Project')
    print('*******************************************************************************')

    mkt_data = clean_data(pd.read_csv('marketing_data.csv'))
    table = PrettyTable()
    print('\n*******************************************************************************')
    print('\nThe following choices are available on the given marketing campaign data')
    print('\n*******************************************************************************')

    table.field_names = field_names
    table.add_row(options)
    print(table)

    choice = input('\nEnter your choice : ')

    if choice == '1':
        analyse_data(mkt_data)
    elif choice == '2':
        do_hypothesis_tests(mkt_data)
    elif choice == '3':
        do_visualisations(mkt_data)
    elif choice == '4':
        plot_correlations(mkt_data)
    elif choice == '5':
        plot_grouped_charts(mkt_data)
    elif choice == '6':
        perform_all_tasks(mkt_data)
    else:
        raise ValueError('The input %c is not recognised' %choice)



if __name__ == '__main__':
    Main()