import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.options.mode.copy_on_write = True
from statsmodels.stats.weightstats import ztest
# example of a one hot encoding
from Common import *

class Tests:

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
        #total_off_store_purchases = data.apply(lambda row: row['TotalAcrossChannels'] - row['NumStorePurchases'],
        #                                       axis=1)
        #data.loc[:, 'OffStorePurchases'] = total_off_store_purchases
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
