"""EzPlot, Class object to generate
commonly used plots for Exploratory Analysis

"""

import sys

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from ggplot import *
from sklearn import ensemble
import xgboost as xgb

# Notes

# ================ Meta ====================
__description__ = 'EzPlot, plots for exploratory analysis'
__version__ = '0.1.0'
__license__ = 'None'
__author__ = 'Eugine Kang (kangeugine@gmail.com)'
__status__ = "Prototype"
__copyright__ = "Copyright 2017"

# python -c 'import sealiondata; sealiondata.package_versions()'
def package_versions():
    print('ezplot         \t', __version__)
    print('python         \t', sys.version[0:5])
    print('numpy          \t', np.__version__)
    print('pandas         \t', pd.__version__)
    print('sns            \t', sns.__version__)

# ================ GLOBAL ====================
color = sns.color_palette()


class EzPlot(object):
    global color
    
    def __init__(self, data):
        self.data = data
        
    def _imputation(self, _data, method='mean'):
        data_copy = _data.copy()
        # Imputation
        if method == 'mean':
            mean_values = data_copy.mean(axis=0)
            data_copy_new = data_copy.fillna(mean_values, inplace=True)
        elif imputation == 'zero':
            data_copy_new = data_copy.fillna(0, inplace=True)
        else:
            data_copy_new = data_copy.fillna(0, inplace=True)
            
        return data_copy_new
        
    def uni_sort_scatter(self, y, size=(8,6)):
        plt.figure(figsize=size)
        plt.scatter(x=np.arange(self.data.shape[0]), y=np.sort(self.data[y].values))
        plt.xlabel('index')
        plt.ylabel(y)
        
    def dist(self, y, size=(12,8)):
        plt.figure(figsize=size)
        sns.distplot(a=self.data[y], bins=100, kde=False)
        plt.xlabel(y, fontsize=12)
        plt.show()
        
    def bar(self, y, size=(12,6)):
        cnt_y = self.data[y].value_counts()

        plt.figure(figsize=size)
        sns.barplot(cnt_y.index, cnt_y.values, alpha=0.8, color=color[3])
        plt.xticks(rotation='vertical')
        plt.xlabel(y, fontsize=12)
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.show()
        
    def count(self, y, size=(12,8)):
        plt.figure(figsize=size)
        sns.countplot(x=y, data=self.data)
        plt.ylabel('Count', fontsize=12)
        plt.xlabel(y, fontsize=12)
        plt.xticks(rotation='vertical')
        _title = ' '.join(['Frequency of', y, 'Count'])
        plt.title(_title, fontsize=15)
        plt.show()
        
    def na_column(self, size=(12,18)):
        missing_df = self.data.isnull().sum(axis=0).reset_index()
        missing_df.columns = ['column_name', 'missing_count']
        # missing_df = missing_df.ix[missing_df['missing_count']>0]
        missing_df = missing_df.sort_values(by='missing_count')
        
        ind = np.arange(missing_df.shape[0])
        width = 0.9
        fig, ax = plt.subplots(figsize=size)
        rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
        ax.set_yticks(ind)
        ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
        ax.set_xlabel("Count of missing values")
        ax.set_title("Number of missing values in each column")
        plt.show()
        
    def joint(self, x, y, size=(12,12)):
        plt.figure(figsize=size)
        sns.jointplot(x=self.data[x].values, y=self.data[y].values, size=10)
        plt.ylabel(x, fontsize=12)
        plt.xlabel(y, fontsize=12)
        plt.show()
        
    def corr(self, y, x=None, imputation='mean', size=(12, 40)):
        '''
        imputation = ['mean', 'zero']
        '''
        corr_data = self.data.copy()
        
        corr_data_new = self._imputation(_data = corr_data, method=imputation)
        
        # Features to Calculate Correlation
        if x:
            pass
        else:
            x =  [col for col in corr_data_new.columns if col not in [y] if corr_data_new[col].dtype in ['float64','int64']]
        
        #print x
        labels = []
        values = []
        for col in x:
            labels.append(col)
            values.append(np.corrcoef(corr_data_new[col].values, corr_data_new[y].values)[0,1])
        corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
        corr_df = corr_df.sort_values(by='corr_values')

        ind = np.arange(len(labels))
        width = 0.9
        fig, ax = plt.subplots(figsize=size)
        rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
        ax.set_yticks(ind)
        ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
        ax.set_xlabel("Correlation coefficient")
        ax.set_title("Correlation coefficient of the variables")
        #autolabel(rects)
        plt.show()
        
        return corr_df
        
    def corr_heatmap(self, x, imputation='mean', size=(8,8)):
        temp_df = self.data[x]
        temp_df_new = self._imputation(_data =temp_df, method=imputation)

        corrmat = temp_df_new.corr(method='spearman')
        
        f, ax = plt.subplots(figsize=size)
        # Draw the heatmap using seaborn
        sns.heatmap(corrmat, vmax=1., square=True)
        plt.title("Important variables correlation map", fontsize=15)
        plt.show()
        
    def box(self, x, y, size=(12,8)):
        plt.figure(figsize=size)
        sns.boxplot(x=x, y=y, data=self.data)
        plt.ylabel(y, fontsize=12)
        plt.xlabel(x, fontsize=12)
        plt.xticks(rotation='vertical')
        _title = ' '.join(['How', y, 'changes with', x])
        plt.title(_title, fontsize=15)
        plt.show()
        
    def violin(self, x, y, size=(12,8)):
        plt.figure(figsize=size)
        sns.violinplot(x=x, y=y, data=train_df)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.show()
    
    def scatter_hue(self, x, y, z, size=(12,8), kind='matplot'):
        '''
        kind = ['matplot', 'ggplot']
        '''
        if kind == 'matplot':
            x = self.data[x].values
            y = self.data[y].values
            z = self.data[z].values

            cmap = sns.cubehelix_palette(as_cmap=True)
            plt.figure(figsize=size)
            f, ax = plt.subplots()
            points = ax.scatter(x, y, c=z, s=50, cmap=cmap)
            f.colorbar(points)
            
        elif kind == 'ggplot':
            print(ggplot(aes(x=x, y=y, color=z), data=train_df) + \
            geom_point(alpha=0.3) + \
            scale_color_gradient(low = 'pink', high = 'blue'))
        
    def pair(self, x, hue=None, imputation='mean'):
        pair_df = self.data[x]
        pair_df = self._imputation(pair_df, imputation)
        
        sns.pairplot(pair_df, hue=hue, kind="reg", palette="husl")
        plt.show()
        
    def importance_extratree(self, do_not_use, y, size=(12,12), imputation='mean'):
        data_df = self._imputation(self.data.copy(), imputation)
        train_y = data_df[y].values
        train_df = data_df.drop(do_not_use, axis=1)
        feat_names = train_df.columns.values

        model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)
        model.fit(train_df, train_y)

        ## plot the importances ##
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        indices = np.argsort(importances)[::-1][:20]

        plt.figure(figsize=size)
        plt.title("Feature importances")
        plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
        plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
        plt.xlim([-1, len(indices)])
        plt.show()
        
    def importance_xgboost(self, do_not_use, y, size=(12,18), imputation='mean'):
        data_df = self._imputation(self.data.copy(), imputation)
        train_y = data_df[y].values
        train_df = data_df.drop(do_not_use, axis=1)
        feat_names = train_df.columns.values
        
        xgb_params = {
            'eta': 0.05,
            'max_depth': 8,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'reg:linear',
            'silent': 1,
            'seed' : 0
        }
        dtrain = xgb.DMatrix(train_df, train_y, feature_names=feat_names)
        model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

        # plot the important features #
        fig, ax = plt.subplots(figsize=(12,18))
        xgb.plot_importance(model, height=0.8, ax=ax)
        plt.show()