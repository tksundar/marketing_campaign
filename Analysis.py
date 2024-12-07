import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.options.mode.copy_on_write = True

# example of a one hot encoding

from Common import *

class Analysis:

    def __init__(self, data):
        self.data = data

    def plot_bar_plot(self,x_values,y_values):
        for i, x in enumerate(x_values):
            fig, axes = plt.subplots(nrows=1, ncols=len(y_values), figsize=(16, 6), layout='constrained')
            title = 'Bar charts of %s grouped by %s ' % (x, hues[i])
            plt.suptitle(title)
            for j, y in enumerate(y_values):
                df = pd.DataFrame(self.data.groupby([x,hues[i]])[y].mean()).reset_index()
                bars = sns.barplot(df, x=x, y=y, hue=hues[i],ax=axes[j])
                bars.legend(loc='upper left', bbox_to_anchor=(1, 1))
                xticks = bars.get_xticklabels()
                bars.set_xticks(bars.get_xticks())
                bars.set_xticklabels(xticks, rotation=90)
            plt.show()

    def grouped_bar_chart(self):
        print('****************************************************************************')
        print('Overall snapshot of marketing data')
        print('****************************************************************************')
        len1 = int(len(dep_vars)/2)
        len2 = int(len(products)/2)
        columns1 = dep_vars[0:len1]
        columns2 = dep_vars[len1:len1*2]
        self.plot_bar_plot(ind_vars,columns1)
        self.plot_bar_plot(ind_vars,columns2)
        prd1 = products[0:len2]
        prd2 = products[len2 :len2*2]
        self.plot_bar_plot(ind_vars,prd1)
        self.plot_bar_plot(ind_vars,prd2)



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
        print('****************************************************************************')
        print('distribution charts before and after outlier treatment')
        print('****************************************************************************')
        for y in dep_vars:
            for x in ind_vars_1:
                self.plot_one_box_and_histogram(x,y)

    def plot_heatmaps(self):
        print('****************************************************************************')
        print('Heatmaps showing various relationships')
        print('****************************************************************************')
        fig_size = (20, 8)
        cmap = 'Paired'
        data = remove_outliers(self.data, dep_vars, 3)

        ind_vars.append('Marital_Status')
        for _, ind_var in enumerate(ind_vars):
            fig, axes = plt.subplots(nrows=1, ncols=len(products), figsize=fig_size, layout='constrained')
            for i, dep_var in enumerate(products):
                df = pd.DataFrame(data.groupby(ind_var)[dep_var].sum()).reset_index()
                p1 = df.pivot_table(columns=ind_var, values=dep_var, aggfunc='sum')
                heatmap = sns.heatmap(p1,annot=True, cmap=cmap, annot_kws={'rotation': 90},ax=axes[i])
                heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
            plt.show()
