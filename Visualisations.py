import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from prettytable import PrettyTable
from scipy.stats import pearsonr

import Common

pd.options.mode.copy_on_write = True

# example of a one hot encoding

from Common import *


def get_color(corr):
    green_threshold = 0.1
    red_threshold = -0.1
    if corr > green_threshold: return 'green'
    if red_threshold < corr < green_threshold: return 'blue'
    if corr < red_threshold: return 'red'




class Visualisations:

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
        fig, ax = plt.subplots(1,1,figsize = (10,6))
        title = 'Top Performer product : %s\nLowest Revenue is for:  %s ' % (top_performer, bottom_performer)
        print('******************************************************************************')
        print('Visualization 1: ', title)
        print(get_table(df))
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
        print(get_table(df))

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x='Age', y='Response', data=df,ax = ax).set_title(
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
        print(get_table(df))

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
        print(get_table(dfp))

        fig,axes = plt.subplots(1,1,figsize= (10,6), layout = 'constrained')
        sns.regplot(x='TotalChildren', y='MeanAcrossProducts', data=dfp
                    ,ax = axes,line_kws={"color": "red"}).set_title(title)
        plt.show()

    def correlation_between_education_and_complaints(self):
        fig, axes = plt.subplots(1, 1, figsize=(10, 6), layout='constrained')

        df = pd.DataFrame(self.data.groupby(['EEducation','Education']).Complain.sum()).reset_index()
        corr,_ = pearsonr(df['EEducation'],df['Complain'])
        #dfs = pd.DataFrame(self.data.pivot_table(columns = ['Education'],values = ['Complain'],aggfunc= 'sum'))
        max_row = pd.DataFrame(df[df['Complain'] == max(df['Complain'])])
        min_row = pd.DataFrame(df[df['Complain'] == min(df['Complain'])])
        qual_max = max_row.iloc[0]['Education']
        qual_min = min_row.iloc[0]['Education']
        qual_string = 'Customers with %s degree complained the most and those with %s degree complained the least '%(qual_max,qual_min)
        corr_string = self.get_correlation(corr)
        title = '%s\nThere is %s between education and complaints\nPearson Correlation Coefficient = %.4f'%(qual_string,corr_string,corr)
        print('********************************************************************')
        print('Visualization 5:')
        print(title)
        print(get_table(df))
        sns.regplot(x='EEducation',y='Complain',data = df,ax=axes).set_title(title)
        plt.show()
    def get_title(self,df,column1,column2):
        tokens = []
        for index, row in df.iterrows():
            tokens.append(str(int(row[column1])))
            tokens.append(' = ')
            tokens.append(row[column2])
            tokens.append('     ')
        return ''.join(tokens)
    def plot_correlations_for(self,dep):
        print('*************************************************************************************')
        print('Spending patterns of various groups across products and channels')
        ind_vars__ = ['EEducation' if element == 'Education' else element for element in ind_vars_1]
        ind_vars__ = ['ECountry' if element == 'Country' else element for element in ind_vars__]

        edn = pd.DataFrame(self.data.groupby(['EEducation', 'Education']).TotalAcrossProducts.sum()).reset_index()
        e_title = self.get_title(edn, 'EEducation', 'Education')
        ectr = pd.DataFrame(self.data.groupby(['ECountry', 'Country']).TotalAcrossProducts.sum()).reset_index()
        c_title = self.get_title(ectr, 'ECountry', 'Country')
        for i, x in enumerate(ind_vars__):
            fig, axes = plt.subplots(nrows=1, ncols=len(dep), figsize=(20, 6), layout='constrained')
            if x == 'EEducation':
                plt.suptitle(e_title)
            elif x == 'ECountry':
                plt.suptitle(c_title)
            for j, y in enumerate(dep):
                df = pd.DataFrame(self.data.groupby(x)[y].sum()).reset_index()
                cor, _ = pearsonr(df[x], df[y])
                sns.regplot(x=x, y=y, data=df, line_kws={"color": get_color(cor)}, ax=axes[j]).set_title(
                    'corr = %.4f' % cor)

        plt.show()
    def correlation_of_spending_patterns(self):
        self.plot_correlations_for(dep_vars)
        self.plot_correlations_for(products)
