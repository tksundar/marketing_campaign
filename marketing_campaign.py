import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.dtypes.common import is_object_dtype
import scipy.stats as stats
from scipy.stats import normaltest
from scipy.stats import pearsonr
from seaborn import reset_orig
from statsmodels.stats.weightstats import ztest

pd.options.mode.copy_on_write = True

# example of a one hot encoding
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def drop_outlier_age_records(data):
    if is_normal(data['Age']):
        outliers = detect_outliers_z(data['Age'])
    else:
        outliers = detect_outliers_iqr(data['Age'])
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


def one_hot_encode(values):
    encoder = OneHotEncoder()
    return encoder.fit_transform(values)


def ordinal_encode(values):
    encoder = OrdinalEncoder()
    return encoder.fit_transform(values)


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
    respectively. Replacing these values with the mode for the column
    '''
    marital_status_mode = pd.DataFrame(mkt_data['Marital_Status'].mode()).iloc[0]['Marital_Status']
    mkt_data['Marital_Status'] = mkt_data.apply(lambda row: get_marital_status(row,marital_status_mode) , axis=1)
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

    # new_mkt_data = mkt_data[mkt_data['Year_Birth'] > 1942]
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
    #drop outlier age records
    new_mkt_data = drop_outlier_age_records(new_mkt_data)
    # check that there are no more missing values
    missing = mkt_data.isna()
    if True in missing.values:
        raise ValueError('Missing values present')
    tasks_completed= ['Stripped all whitespace',
                      'Replaced nan values with appropriate values',
                      'Converted all date columns to pandas datetime dtype',
                      'Cleaned Marital_Status column by replacing bad values with column mode value',
                      'Cleaned the Income column by \n'
                       '\ta)Removing $ sign from the entries and converting the dtype to float64 so that mathematical operations are possible\n'
                       '\tb)Replacing nan entries(24 of them) with suitable mean values of income based on marital status and education level of the customer\n',
                      'Removed outlier records for age',
                      'Added TotalAcrossProducts, TotalAcrossChannels and Age columns',
                      'Outlier Age records dropped']
    print('Cleaning and filling missing values complete.')
    print('Cleaning performed the following tasks')
    print('------------------------------------------------------------------')
    for i,v in enumerate(tasks_completed):
        print('%d. %s' %(i,v))

    print('data shape after cleaning : ', new_mkt_data.shape)
    return new_mkt_data


class Analysis():

    def __init__(self, data):
        self.data = data

    def plot_bar_plot(self,data,x,y,hue):
        fig, ax = plt.subplots(figsize=(16, 8), layout='constrained')
        sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax)
        plt.ylabel('Mean Spends across all products')
        plt.title('Snapshot of total expenditure by various categories by %s and %s' %(x,hue))
        plt.show()

    def grouped_bar_chart(self):
        columns = ['Country', 'Education', 'Marital_Status']
        df = pd.DataFrame(self.data.groupby(columns).TotalAcrossProducts.mean()).reset_index()
        self.plot_bar_plot(df,x=columns[0],y='TotalAcrossProducts',hue=columns[1])
        self.plot_bar_plot(df,x=columns[0],y='TotalAcrossProducts',hue=columns[2])
        self.plot_bar_plot(df,x=columns[1],y='TotalAcrossProducts',hue=columns[2])

    def plot_one_box_and_histogram(self,ind_var,dep_var):
        fig_size = (16, 10)
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=fig_size, layout='constrained')
        df1 = self.data.loc[:, [ind_var, dep_var]].sort_values(by=ind_var)
        cleaned_data = remove_outliers(self.data,[dep_var])
        df2 = cleaned_data.loc[:, [ind_var, dep_var]].sort_values(by=ind_var)
        str1 = f'Z score (threshold = %.1f) treatment for outliers,\nReplacement strategy : median' % threshold
        str2 = f'IQR(factor = %.1f) treatment for outliers,\nReplacement strategy : median' % factor
        fn = lambda v: str1 if is_normal(v) else str2
        sns.histplot(df1, x=ind_var, kde=True, ax=axes[0])
        sns.boxplot(y=dep_var, data=df1, ax=axes[1]).set_title('Before %s ' % fn(df1[dep_var]))
        sns.boxplot(y=dep_var, data=df2, ax=axes[2]).set_title('After %s ' % fn(df2[dep_var]))
        plt.show()

    def plot_box_and_histogram(self):
        for y in dep_vars:
            for x in ind_vars:
                self.plot_one_box_and_histogram(x,y)

    def plot_heatmaps(self):
        fig_size = (12, 8)
        cmap = 'tab20'
        data = remove_outliers(self.data, dep_vars, 3)
        if 'Education' in ind_vars:
            encoded = ordinal_encode(pd.DataFrame(data['Education']))
            data.loc[:, 'Education'] = encoded
        for _, ind_var in enumerate(ind_vars):
            fig, axes = plt.subplots(nrows=1, ncols=len(dep_vars), figsize=fig_size,layout='constrained')
            for i, dep_var in enumerate(dep_vars):
                df = pd.DataFrame(data.groupby(ind_var)[dep_var].sum()).reset_index()
                sns.heatmap(df, cmap=cmap, ax=axes[i])
            plt.show()


class Tests():


    def __init__(self,data):
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

    def plot_purchase_means(self,labels, H0, H1, pop_mean, data_with_kids, data_without_kids, ax):
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

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
        self.plot_purchase_means(labels, H0, H1, pop_mean,
                                  data_with_kids,
                                  data_without_kids,
                                  axes[0])
        axes[0].set_ylabel('Web Purchases')

        H0 = 'Customers with kids prefer  off store purchases'
        H1 = 'FTR - Customers with kids do not prefer off store purchases'
        off_store_purchases = data.apply(lambda row: row['TotalAcrossChannels'] - row['NumStorePurchases'], axis=1)
        data.loc[:, 'OffStorePurchases'] = off_store_purchases
        off_store_with_kids = data[data['Kidhome'] > 0].loc[:, ['Kidhome', 'OffStorePurchases']]
        off_store_without_kids = data[data['Kidhome'] == 0].loc[:, ['Kidhome', 'OffStorePurchases']]
        pop_mean = data['OffStorePurchases'].mean()
        labels = ['With Kids', 'without kids']
        Tests.plot_purchase_means(labels, H0, H1, pop_mean,
                                  off_store_with_kids['OffStorePurchases'],
                                  off_store_without_kids['OffStorePurchases'],
                                  axes[1])
        axes[1].set_ylabel('Off Store Purchases')
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
        total_off_store_purchases = data.apply(lambda row: row['TotalAcrossChannels'] - row['NumStorePurchases'],
                                               axis=1)
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
        H0 = 'There is no difference in US purchase volumes compared to other countries'
        H1 = 'US  outperforms the rest of the countries in purchase volumes'
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
        fig, a = plt.subplots(1, 1, figsize=(10, 6))
        sns.barplot(x='Country', y='MeanAcrossProducts', hue='Country', data=by_country, ax=a)
        plot_title = '%s\nPopulation mean = %.4f,\n US mean = %.4f,\n p value = %.4f,\nalpha %.4f' % (
            title, pop_mean, us_mean, p, alpha)
        plt.suptitle(plot_title)
        plt.tight_layout()
        plt.show()


class Visualise():

    def __init__(self, data):
        self.data = data

    def color_max_min(self, ax, values):
        for bar in ax.patches:
            if bar.get_height() == max(values):
                bar.set_color('skyblue')
            elif bar.get_height() == min(values):
                bar.set_color('grey')
            else:
                bar.set_color('orange')

    def get_correlation(self, corr):
        weak_threshold = .3
        medium_threshold = .6

        if abs(corr) < weak_threshold:
            if corr < 0:
                return 'weak  %s correlation' % 'negative'
            else:
                return 'weak  % correlation' % 'positive'

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
        fig, ax = plt.subplots(1,1,figsize = (10,6))
        title = 'Top Performer product : %s\nLowest Revenue is for:  %s ' % (top_performer, bottom_performer)
        print('******************************************************************************')
        print('Visualization 1: ', title)
        print(df)

        sns.barplot(x='Products', y='Amount', hue='Products', data=df,ax= ax).set_title(title)
        self.color_max_min(ax, values)
        plt.xticks(rotation=45)
        plt.show()

    def correlation_customer_age_and_campaign(self):
        data = self.data
        correlation, _ = pearsonr(data['Age'], data['Response'])
        title = self.get_correlation(correlation)
        df = pd.DataFrame(data.groupby('Age').Response.sum()).reset_index()
        print('*****************************************************************************')
        print('Visualization 2: ', title)
        print(df)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Age', y='Response', data=df,ax = ax).set_title(
            'Correlation between age and campaign response.\n%s.  Pearsons Correlation Coefficient = %.4f' % (title, correlation))
        plt.show()

    def country_with_highest_campaign_response(self):
        df = pd.DataFrame(self.data.groupby(['Country']).Response.sum()).reset_index()
        max_response = df[df['Response'] == max(df['Response'])]
        max_response_country = max_response['Country'].tolist()[0]
        max_response_value = max_response['Response'].tolist()[0]
        fig, ax = plt.subplots(figsize = (10,6))
        title = 'Country with max campaign response acceptance\nCountry : %s  Accepted count = %d' % (max_response_country, max_response_value)
        print('**************************************************************************')
        print('Visualization 3: ', title)
        print(df)

        sns.barplot(x='Country', y='Response', data=df).set_title(title)
        self.color_max_min(ax, df['Response'].tolist())
        plt.show()

    def correlation_num_kids_and_total_spend(self):
        # new_mkt_data.loc[:, 'TotalAcrossProducts'] = total_across_products
        total_kids = self.data.apply(lambda  row: row['Kidhome']+row['Teenhome'],axis = 1)
        self.data.loc[:,'TotalChildren'] = total_kids
        dfp = pd.DataFrame(self.data.groupby('TotalChildren').TotalAcrossProducts.mean()).reset_index()
        dfp.rename(columns = {'TotalAcrossProducts':'MeanAcrossProducts'},inplace=True)
        corr, _ = pearsonr(dfp['TotalChildren'], dfp['MeanAcrossProducts'])

        title = 'Correlation between total children and expenditure\n%s, Pearson correlation coefficient = %.4f' % (self.get_correlation(corr), corr)
        print('******************************************************************************')
        print('Visualization 4: ', title)
        print(dfp)

        fig,axes = plt.subplots(1,1,figsize= (10,6), layout = 'constrained')
        sns.regplot(x='TotalChildren', y='MeanAcrossProducts', data=dfp
                    ,ax = axes,line_kws={"color": "red"}).set_title(title)
        plt.show()

    def correlation_between_education_and_complaints(self):
        styles = [{'selector': '*', 'props': [
            ('font_family', 'Mono'),
            ('font-size', '15px'),
            ('margin', '15px auto'),
            ('border', '2px solid #ccc'),
            ('border-bottom', '2px solid #00eeee')
        ]}]
        fig, axes = plt.subplots(1, 1, figsize=(10, 6), layout='constrained')
        encoded = ordinal_encode(pd.DataFrame(self.data['Education']))
        self.data.loc[:, 'EEducation'] = encoded
        df = pd.DataFrame(self.data.groupby(['EEducation','Education']).Complain.sum()).reset_index()
        corr,_ = pearsonr(df['EEducation'],df['Complain'])
        dfs = pd.DataFrame(self.data.pivot_table(columns = ['Education'],values = ['Complain'],aggfunc= 'sum'))

        corr_string = self.get_correlation(corr)
        title = 'There is %s between education and complaints\nPearson Correlation Coefficient = %.4f'%(corr_string,corr)
        print('********************************************************************')
        print('Visualization 5:')
        print(title)

        print(dfs)
        sns.regplot(x='EEducation',y='Complain',data = df,ax=axes).set_title(title)


        plt.show()


def analyse_data(mkt_data):
    print('*******************************************************************************')
    print('Starting basis data analysis for preliminary insights')
    print('*******************************************************************************')
    global factor
   # data = remove_outliers(mkt_data.copy(deep=True), dep_vars)
    analysis = Analysis(mkt_data)
    analysis.grouped_bar_chart()
    analysis.plot_box_and_histogram()
    analysis.plot_heatmaps()



def do_hypothesis_tests(data):
    print('*******************************************************************************')
    print('Starting hypothesis testing')
    print('*******************************************************************************')
    Tests.test_older_individuals_rely_on_store_shopping(data)
    Tests.test_kids_influence_online_shopping(data)
    Tests.test_cannibalization_by_off_store_sales_channels(data)
    Tests.test_us_outperforms_purchases(data)


def do_visualisations(data):
    print('*******************************************************************************')
    print('Starting Visualisations for correlations')
    print('*******************************************************************************')
    visualise = Visualise(remove_outliers(mkt_data, products))
    visualise.top_performer()
    visualise.correlation_customer_age_and_campaign()
    visualise.country_with_highest_campaign_response()
    visualise.correlation_num_kids_and_total_spend()
    visualise.correlation_between_education_and_complaints()


threshold = 3.0  # for outlier removal using z score for normal distributions
factor = 1.5  # for outlier removal using IQR interval for skewed distributions
products = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
prod_labels = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
dep_vars = ['TotalAcrossProducts', 'TotalAcrossChannels', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
            'NumStorePurchases']
ind_vars = ['Age', 'Education', 'Kidhome']

menu = ['Analyse Data','Do hypothesis Tests','Do Correlations','Perform All Tasks']

def perform_all_tasks(mkt_data):
    analyse_data(mkt_data)
    do_hypothesis_tests(mkt_data)
    do_visualisations(mkt_data)

if __name__ == '__main__':
    print('*******************************************************************************')
    print('Marketing Campaign Project')
    print('*******************************************************************************')
    mkt_data = clean_data(pd.read_csv('marketing_data.csv'))
    for i,v in enumerate(menu):
        print('%d %s' %(i+1, v))

    choice = input('\nEnter your choice : ')

    if choice == '1':
        analyse_data(mkt_data)
    elif choice == '2':
        do_hypothesis_tests(mkt_data)
    elif choice == '3':
        do_visualisations(mkt_data)
    elif choice == '4':
        perform_all_tasks(mkt_data)
    else:
        raise ValueError('Illegal input')






