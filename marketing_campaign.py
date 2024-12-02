import datetime
from cgitb import reset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pandas.core.dtypes.common import is_object_dtype

from statsmodels.stats.weightstats import ztest
from scipy.stats import normaltest
from scipy.stats import pearsonr

pd.options.mode.copy_on_write = True



# example of a one hot encoding
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def is_normal(data):
    """
    checks if the data is normally distributed
    """

    alpha = 0.005
    _,p = normaltest(data)
    if p > alpha:
        return True
    return False

def one_hot_encode(values):
    encoder = OneHotEncoder()
    return encoder.fit_transform(values)


def ordinal_encode(values):
    encoder = OrdinalEncoder()
    return encoder.fit_transform(values)


def get_income(row, data):
    key = (row['Marital_Status'], row['Education'])
    return data.get(key)


def get_marital_status(row):
    if row['Marital_Status'] == 'Alone' or row['Marital_Status'] == 'YOLO' or row['Marital_Status'] == 'Absurd':
        return 'Single'
    else:
        return row['Marital_Status']


def print_test_header(H0, H1):
    print('*************************************************************************************')
    print('H0', H0)
    print('H1', H1)
    print('*************************************************************************************')
def detect_outliers_z(data,threshold=3.0):
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

def remove_outliers(data,outlier_columns,threshold = 3, factor = 1.5):
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
          outliers = detect_outliers_iqr(array,factor)
          value = np.median(array)
        else:
            outliers = detect_outliers_z(array.data, threshold)
            value = np.mean(array)
        for i in outliers:
           #for every i(outlier) in array , replace it by value else retain the same array element
           array = np.where(array == i, value, array)
           data[column] = np.asarray(array)

    return data

def clean_data(mkt_data):
    """
     Cleans the input by doing the following
     1. Strips all whitespace
     2. replaces nan values with appropriate values
     3. Converts all date columns to pandas datetime dtype
     4. Cleans Marital_Status column by replacing bad values with sensible ones
     5. Cleans the Income column by
        a)Removing $ sign from the entries and converting the dtype to float64 so that mathematical operations are possible
        b)Replacing nan entries(24 of them) with suitable mean values of income based on maritcal status and education level of the customer
    :param mkt_data: DataFrame
    :return:  mkt_data: DataFrane
    """
    print('data shape before cleaning : ',mkt_data.shape)
    mkt_data.drop_duplicates(keep='first',inplace=True)
    for col in mkt_data.columns:
        if ' ' in col:
            mkt_data.rename(columns={col: col.strip()}, inplace=True)
    for col in mkt_data.columns:
        if is_object_dtype(mkt_data[col]):
            mkt_data[col] = mkt_data[col].str.strip()

    mkt_data['Income'] = mkt_data['Income'].str.replace('$', '').str.replace(',', '', regex=True).astype(float)
    mkt_data['Dt_Customer'] = pd.to_datetime(mkt_data['Dt_Customer'], format="%m/%d/%y")
    '''
    The data contains the following marital statuses
    ['Divorced' 'Single' 'Married' 'Together' 'Widow' 'YOLO' 'Alone' 'Absurd']
    The first four are valid statuses (Together could mean a livin relationship)
    The last three are not acceptable. However their frequency count is very lo at 3,2,2 
    respectively. Alone could be considered as single and i am putting the others also
    in this category
    '''
    mkt_data['Marital_Status'] = mkt_data.apply(lambda row: get_marital_status(row), axis=1)
    income_by_ms_and_eduction = pd.DataFrame(mkt_data.groupby(['Marital_Status', 'Education']).Income.mean())
    '''
    incomes is a dictionary where key is a tuple and value is the mean for that combination like -> ('Married', 'Master'): 53286.02898550725 
    We will use this dictionary to replace nan income cells with appropriate values. This is based on the problem statement
    Another way is to use country and education to replace the income values
    '''
    incomes = income_by_ms_and_eduction.to_dict().get('Income')

    '''
    We fill the missing values in a row by the mean value of income for that row's marital_status and eduction level.
    we use the incomes dictionary for lookup
    '''
    mkt_data.loc[:, 'Income'] = mkt_data.apply(
        lambda row: get_income(row, incomes) if np.isnan(row.loc['Income']) else row.loc['Income'], axis=1)

    '''
    we create thee more columns to hold the total sales across product lines , across sales channels and 
    a column for age (calculated from year of birth)
    '''

    #new_mkt_data = mkt_data[mkt_data['Year_Birth'] > 1942]
    new_mkt_data = mkt_data.copy()
    total_across_products = new_mkt_data.apply(
        lambda row: row['MntWines'] + row['MntFruits'] + row['MntMeatProducts'] + row['MntFishProducts'] + row[
            'MntGoldProds'], axis=1)
    total_across_channels = new_mkt_data.apply(
        lambda row: row['NumDealsPurchases'] + row['NumWebPurchases'] + row['NumCatalogPurchases'] + row[
            'NumStorePurchases'] + row['MntGoldProds'], axis=1)
    new_mkt_data.loc[:, 'TotalAcrossProducts'] = total_across_products
    new_mkt_data.loc[:, 'TotalAcrossChannels'] = total_across_channels

    current_year = datetime.datetime.now().year
    new_mkt_data.loc[:, 'Age'] = new_mkt_data.apply(lambda row: current_year - row['Year_Birth'], axis=1)
    # total_kids = new_mkt_data.apply(lambda row: row['Kidhome'] + row['Teenhome'], axis=1)
    # new_mkt_data.loc[:, 'TotalKids'] = total_kids

    # check that there are no more missing values
    missing = mkt_data.isna()
    if True in missing.values:
        raise ValueError('Missing values present')
    print('Cleaning and filling missing values complete. Three new columns viz., TotalAcrossProducts, TotalAcrossChannels and Age have been added')
    print('data shape before cleaning : ',new_mkt_data.shape)
    return new_mkt_data

class Analysis():
    def plot_box_and_histogram(raw_data,cleaned_data, ind_vars,dep_vars):
        fig_size = (16, 10)
        for  y in dep_vars:
            for x in ind_vars:
                fig, axes = plt.subplots(ncols=3, nrows=1, figsize=fig_size, layout='constrained')
                df1 = raw_data.loc[:,[x,y]].sort_values(by=x)
                df2 = cleaned_data.loc[:, [x, y]].sort_values(by=x)
                str1 = f'Z score (threshold = %.1f) treatment for outliers,\nReplacement strategy : median' % threshold
                str2 = f'IQR(factor = %.1f) treatment for outliers,\nReplacement strategy : median' % factor
                fn = lambda v: str1 if is_normal(v) else str2
                sns.histplot(df1, x=x, kde=True, ax=axes[0])
                sns.boxplot(y=y, data=df1, ax=axes[1]).set_title('Before %s ' % fn(df1[y]))
                sns.boxplot(y=y, data=df2, ax=axes[2]).set_title('After %s ' % fn(df2[y]))
                plt.show()




    def plot_box_plot(mkt_data, x_axis=None, y_axis=None,title=None):
        fig_size = (8, 12)
        x = np.arange(10)
        for i,y in enumerate(y_axis):
            fig, axes = plt.subplots(ncols=len(x_axis), nrows=1, figsize=fig_size,layout='constrained')
            for j, x in enumerate(x_axis):
              sns.boxplot(x=x, y=y, data=mkt_data, ax=axes[j])
              if x == 'Age':
                   xticks = np.arange(25,5,100)
                   xlabels=[x for x in xticks]
                   axes[j].set_xticks(xticks, labels=xlabels,rotation=45)
            plt.suptitle(title)
            plt.tight_layout()
            plt.show()


    def plot_heatmaps(mkt_data,ind_vars,dep_vars):
        fig_size = (12,8)
        cmap = 'tab20'
        data = remove_outliers(mkt_data, dep_vars, 3)
        if 'Education' in ind_vars:
            encoded = ordinal_encode(pd.DataFrame(data['Education']))
            data.loc[:, 'Education'] = encoded
        for _,ind_var in enumerate(ind_vars):
            fig, axes = plt.subplots(nrows=1, ncols=len(dep_vars), figsize=fig_size)
            for i,dep_var in enumerate(dep_vars):
              df = pd.DataFrame(mkt_data.groupby(ind_var)[dep_var].sum()).reset_index()
              sns.heatmap(df,cmap=cmap,ax=axes[i])
            plt.tight_layout()
            plt.show()

class Tests():

    def test_older_individuals_rely_on_store_shopping(data):
        """
       Here we use z test to confirm or reject the null hypothesis.
        H0 = 'There is no difference in store shopping pattern between older and younger individuals'
        H1 = 'Older individuals rely on store shopping compared to younger individuals'
       """
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

    def plot_purchase_means(labels,H0,H1,pop_mean,data_with_kids,data_without_kids,ax):
        alpha = 0.001
        _, p = ztest(data_without_kids,data_with_kids)
        if p > alpha:
            title = H0
        else:
            title = H1
        with_kids_mean = data_with_kids.mean()
        without_kids_mean = data_without_kids.mean()
        values = [with_kids_mean,without_kids_mean]
        sns.barplot(y=values, x=labels, ax=ax)
        t = ('%s\np value = %.5f\nalpha = %.4f, \nPopulation mean=%.4f,\n with_kids_mean = %.4f,\nwithout_kids_mean=%.4f\n'
             % (title,p,alpha,pop_mean,with_kids_mean,without_kids_mean))
        ax.set_title(t)
        plt.tight_layout()

    def test_kids_influence_online_shopping(data):
        """
        Does a Z test on the sample with kids(kids+teens) and the sample with no kids
        """
        H0 = 'Customers with kids prefer  web purchases'
        H1 = 'Customers with kids do not prefer web  purchases'

        print_test_header(H0, H1)
        num_web_purchases_with_kids = data[data['Kidhome'] > 0].loc[:, ['Kidhome', 'NumWebPurchases']]
        num_web_purchases_without_kids = data[data['Kidhome'] == 0].loc[:, ['Kidhome', 'NumWebPurchases']]
        pop_mean = data['NumWebPurchases'].mean()
        data_with_kids =  num_web_purchases_with_kids['NumWebPurchases']
        data_without_kids = num_web_purchases_without_kids['NumWebPurchases']
        labels = ['With Kids', 'without kids']

        fig , axes = plt.subplots(nrows=1,ncols=2,figsize=(10,8))
        Tests.plot_purchase_means(labels, H0, H1,pop_mean,
                             data_with_kids,
                             data_without_kids,
                             axes[0] )
        axes[0].set_ylabel('Web Purchases')

        H0 = 'Customers with kids prefer  off store purchases'
        H1 = 'FTR - Customers with kids do not prefer off store purchases'
        off_store_purchases = data.apply(lambda row: row['TotalAcrossChannels'] - row['NumStorePurchases'], axis=1)
        data.loc[:, 'OffStorePurchases'] = off_store_purchases
        off_store_with_kids = data[data['Kidhome'] > 0].loc[:, ['Kidhome', 'OffStorePurchases']]
        off_store_without_kids = data[data['Kidhome'] == 0].loc[:, ['Kidhome', 'OffStorePurchases']]
        pop_mean = data['OffStorePurchases'].mean()
        labels = ['With Kids', 'without kids']
        Tests.plot_purchase_means(labels, H0, H1,pop_mean,
                             off_store_with_kids['OffStorePurchases'],
                             off_store_without_kids['OffStorePurchases'],
                             axes[1])
        axes[1].set_ylabel('Off Store Purchases')
        plt.show()


    def test_cannibalization_by_off_store_sales_channels(data):
        """
        Assumption : All channels other than store purchases( such as web, deals or catalogue purchases) are treated as off store sales
        This may not be true as discount(deal) and catalogue sales can be from the store too.
        """
        H0 = 'Off Store channels do not cannibalize store purchases'
        H1 = 'Off store channels cannibalize store purchases'
        print_test_header(H0, H1)
        alpha = 0.001
        total_off_store_purchases = data.apply(lambda row: row['TotalAcrossChannels'] - row['NumStorePurchases'], axis=1)
        data.loc[:, 'OffStorePurchases'] = total_off_store_purchases
        _, p = ztest(total_off_store_purchases, data['NumStorePurchases'])
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
        plt.suptitle("%s\n store_purchases_mean = %.4f , \noff_store_purchases_mean = %.4f ,\n p value = %.10f,\nalpha = %.4f"
                     % (title, store_purchases_mean, off_store_purchases_mean, p, alpha))
        plt.tight_layout()
        plt.show()


    def test_us_outperforms_purchases(data):
        """
        H0 : US doesn't outperform the rest of the countries in purchase volumes
        H1 : US  outperforms the rest of the countries in purchase volumes
        :param data:
        :return:
        """
        H0 = 'There is no difference in US purchase volumes compared to other countries'
        H1 = 'US  outperforms the rest of the countries in purchase volumes'
        print_test_header(H0,H1)
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
        fig, a = plt.subplots(1, 1, figsize=(10, 6))
        sns.barplot(x='Country', y='MeanAcrossProducts', hue='Country', data=by_country, ax=a)
        plot_title = '%s\nPopulation mean = %.4f,\n US mean = %.4f,\n p value = %.4f,\nalpha %.4f' % (
        title, pop_mean, us_mean, p, alpha)
        plt.suptitle(plot_title)
        plt.tight_layout()
        plt.show()
class Visualise():

    def __init__(self,data):
        self.data= data

    def color_max_min(self,ax,values):
        for bar in ax.patches:
            if bar.get_height() == max(values):
                bar.set_color('skyblue')
            elif bar.get_height() == min(values):
                bar.set_color('grey')
            else:
                bar.set_color('orange')

    def get_correlation(self,corr):
        weak_threshold = .3
        medium_threshold = .6

        if abs(corr) < weak_threshold:
            if corr < 0:
               return 'weak  %s correlation' % 'negative'
            else:
               return 'weak  % correlation' % 'positive'

        elif  weak_threshold < abs(corr) < medium_threshold:
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
        data = {'Products': prod_labels,'Amount': values}
        df = pd.DataFrame(data)
        top_performer = prod_labels[values.index(max(values))]
        bottom_performer = prod_labels[values.index(min(values))]
        fig,ax = plt.subplots()
        title = 'Top Performer is %s Lowest Revenue is for %s ' %(top_performer,bottom_performer)
        print('Visualization 1: ',title)
        sns.barplot(x= 'Products', y = 'Amount' , hue = 'Products' ,data= df ).set_title(title)
        self.color_max_min(ax,values)
        plt.xticks(rotation=45)
        plt.show()
    def correlation_customer_age_and_campaign(self):
        if is_normal(self.data['Age']):
            outliers = detect_outliers_z(self.data['Age'])
            print('normal')
        else:
            outliers = detect_outliers_iqr(self.data['Age'])
        index_names = self.data[(self.data['Age'] >= min(outliers))].index
        data = self.data.drop(index_names)
        correlation,_ = pearsonr(data['Age'],data['Response'])
        title = self.get_correlation(correlation)
        df = pd.DataFrame(data.groupby('Age').Response.sum()).reset_index()
        print('Visualization 2: ', title)
        sns.scatterplot(x='Age',y='Response' , data = df).set_title('%s.  Pearsons Correlation Coefficient = %.4f' %(title,correlation))
        plt.show()

    def country_with_highest_campaign_response(self):
        df = pd.DataFrame(self.data.groupby(['Country']).Response.sum()).reset_index()
        max_response = df[df['Response'] == max(df['Response'])]
        max_response_country = max_response['Country'].tolist()[0]
        max_response_value = max_response['Response'].tolist()[0]
        fig, ax = plt.subplots()
        title = 'Max Campaign accepted by %s.Number of acceptance = %.1f'%(max_response_country,max_response_value)
        print('Visualization 3: ', title)
        sns.barplot(x='Country', y='Response' , data = df).set_title(title)
        self.color_max_min(ax,df['Response'].tolist())
        plt.show()

    def correlation_num_kids_and_total_spend(self):
        dfp = pd.DataFrame(self.data.groupby('Kidhome').TotalAcrossProducts.sum()).reset_index()
        corr,_ = pearsonr(dfp['Kidhome'],dfp['TotalAcrossProducts'])
        title = 'There is %s between kids and expenditure . Pearson correlation coefficient = %.4f' %(self.get_correlation(corr),corr)
        print('Visualization 4: ', title)
        sns.scatterplot(x='Kidhome',y='TotalAcrossProducts', data = dfp).set_title(title)
        plt.show()


def analyse_data(mkt_data, x_axes=None, categories=None):
    global factor
    data  = remove_outliers(mkt_data.copy(deep = True),categories)
    Analysis.plot_box_and_histogram(mkt_data,data,x_axes,categories)
    Analysis.plot_heatmaps(data,x_axes,categories)

def do_hypothesis_tests(data):
    Tests.test_older_individuals_rely_on_store_shopping(data)
    Tests.test_kids_influence_online_shopping(data)
    Tests.test_cannibalization_by_off_store_sales_channels(data)
    Tests.test_us_outperforms_purchases(data)

def do_visualisations(data):
    visualise = Visualise(remove_outliers(mkt_data, products))
    visualise.top_performer()
    visualise.correlation_customer_age_and_campaign()
    visualise.country_with_highest_campaign_response()
    visualise.correlation_num_kids_and_total_spend()



threshold = 3.0 # for outlier removal using z score for normal distributions
factor = 1.5 # for outlier removal using IQR interval for skewed distributions
products = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
prod_labels = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
dep_vars = ['TotalAcrossProducts', 'TotalAcrossChannels', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
ind_vars = ['Age', 'Education', 'Kidhome']

if __name__ == '__main__':
    mkt_data = clean_data(pd.read_csv('marketing_data.csv'))
    # print('*******************************************************************************')
    # print('Starting basis data analysis for preliminary insights')
    # print('*******************************************************************************')
    #
    # analyse_data(mkt_data, x_axes=ind_vars, categories=dep_vars)
    # print('*******************************************************************************')
    # print('Starting hypothesis testing')
    # print('*******************************************************************************')
    # do_hypothesis_tests(mkt_data)
    print('*******************************************************************************')
    print('Starting Visualisations for correlations')
    print('*******************************************************************************')
    do_visualisations(mkt_data)




