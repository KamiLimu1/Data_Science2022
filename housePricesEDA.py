# %%
pip install missingno

# %%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
import missingno as msno

# %%
pip install seaborn --upgrade

# %%
#training and testing model
train_df = pd.read_csv(r'C:\Users\lenovo\PycharmProjects\housePriceEDA\train.csv')
test_df = pd.read_csv(r'C:\Users\lenovo\PycharmProjects\housePriceEDA\test.csv')

# %%
train_df.describe()

# %%
train_df.head()

# %%
train_df.tail()

# %%
#rows and columns in the data frames
train_df.shape, test_df.shape

# %%
#data types
train_df.dtypes

# %%
#select the numeric columns from the train data frame and return their column names
numeric_features = train_df.select_dtypes(include=[np.number])
numeric_features.columns
numeric_features.shape


# %%
#select the categorical columns(string,object, category) from the train data frame and return their column names
categorical_features = train_df.select_dtypes(include=[object])
categorical_features.columns
categorical_features.shape

# %%
#shows the missing values in a random sample of 250 rows from the train df
msno.matrix(train_df.sample(250))

# %%
#how strongly the presence or absence of one variable affects the presence of another
#missing values shown in red and non-missing values shown in blue.
msno.heatmap(train_df)

# %%
msno.bar(train_df.sample(1000))

# %%
#the bigger the distance between two links,the bigger the difference in terms of the features
msno.dendrogram(df = train_df)

# %%
#Droping unwanted columns
train_df1 = train_df.drop(['Alley', 'Neighborhood', 'Exterior1st', 'Exterior2nd', 'FireplaceQu', 'PoolQC', 
                          'Fence', 'MiscFeature', 'Utilities', 'ExterCond', 'Condition2', 'HouseStyle', 'RoofMatl', 
                           'Exterior1st', 'Exterior2nd', 'Heating', 'Electrical', 'GarageQual', 'PoolQC',
                           'MiscFeature'],axis = 1)

train_df1.shape        
sns.histplot(train_df1['SalePrice'],bins=20)                   

# %%
#symmetry and peakedness
#train_df.skew(), train_df.kurt()
numeric_features.skew(), numeric_features.kurt()

# %%
# how the skew and kurtosis of the data differ for each different distributions
#displays a histogram of the "SalePrice" column in the train_df, with a fitted Johnson SU distribution.
y = train_df['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False,fit=st.johnsonsu)

#histogram of the same data with a fitted normal distribution
plt.figure(2);plt.title('Normal')
sns.distplot(y,kde = False,fit = st.norm)

#histogram of the data with a fitted lognormal distribution
plt.figure(3);plt.title('Log Normal')
sns.distplot(y,kde=False,fit=st.lognorm)

# %%
sns.distplot(train_df.skew(),color ='blue',axlabel='Skewness')

# %%
plt.figure(figsize= (12,8))
sns.distplot(train_df.kurt(),color='r',axlabel='Kurtosis',norm_hist=False,kde=True,rug=False)
plt.show()

# %%
plt.hist(train_df['SalePrice'], orientation='vertical',histtype='bar',color='blue')
plt.show()

# %%
#normalize skewed data and make it more symmetrical for easier analysis and model data
target = np.log(train_df['SalePrice'])
target.skew()
plt.hist(target,color='Blue')

# %%
#relationships between the "SalePrice" data and the other numeric columns in the numeric_features
#values range from -1 to 1, with 1 indicating a perfect positive correlation and -1 indicating a perfect negative correlation
correlation = numeric_features.corr()
print(correlation['SalePrice'].sort_values(ascending=False),'\n')

# %%
f, ax = plt.subplots(figsize = (14,12))
plt.title('Correlation of Numeric Features with Sale Price',y=1,size =16)
sns.heatmap(correlation,square=True,vmax=0.8)

# %%
# Zoomed heatmap
k = 11
cols = correlation.nlargest(k, 'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(train_df[cols].values.T)
f, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    cm, 
    vmax=.8, 
    linewidths=0.01,
    square=True, 
    annot=True, 
    cmap='viridis',
    linecolor='white',
    xticklabels=cols.values,
    annot_kws={
          'size': 12
          },
     yticklabels=cols.values
     )


# %%
sns.set()
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
sns.pairplot(train_df[columns],height=2,kind='scatter',diag_kind='kde')
plt.show()

# %%
#creating a figure with multiple subplots & generating a scatter plot of the "SalePrice" and "X" data, with a fitted regression line.

fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize =(14,10))

OverallQual_scatter_plot = pd.concat([train_df['SalePrice'],train_df['OverallQual']],axis=1)
sns.regplot(x='OverallQual', y='SalePrice', data=OverallQual_scatter_plot, scatter=True,fit_reg=True,ax=ax1,color='blue')
 
TotalBsmtSF_scatter_plot = pd.concat([train_df['SalePrice'], train_df['TotalBsmtSF']],axis=1)
sns.regplot(x='TotalBsmtSF', y='SalePrice', data=TotalBsmtSF_scatter_plot,scatter=True,fit_reg=True,ax=ax2, color= 'orange')

GrLivArea_scatter_plot = pd.concat([train_df['SalePrice'],train_df['GrLivArea']],axis=1)
sns.regplot(x='GrLivArea', y='SalePrice', data=GrLivArea_scatter_plot, scatter=True, fit_reg=True, ax=ax3, color= 'Green')

GarageArea_scatter_plot = pd.concat([train_df['SalePrice'], train_df['GarageArea']], axis=1)
sns.regplot(x='GarageArea', y='SalePrice', data=GarageArea_scatter_plot,scatter=True,fit_reg=True, ax=ax4, color= 'red')

FullBath_scatter_plot = pd.concat([train_df['SalePrice'], train_df['FullBath']], axis=1)
sns.regplot(x='FullBath', y='SalePrice', data=FullBath_scatter_plot,scatter=True,fit_reg=True,ax = ax5, color='Purple')

YearBuilt_scatter_plot = pd.concat([train_df['SalePrice'], train_df['YearBuilt']], axis=1)
sns.regplot(x='YearBuilt', y='SalePrice', data=YearBuilt_scatter_plot,scatter=True,fit_reg=True,ax=ax6,color='brown')

YearRemodAdd_scatter_plot = pd.concat([train_df['SalePrice'], train_df['YearRemodAdd']], axis=1)
YearRemodAdd_scatter_plot.plot.scatter('YearRemodAdd', 'SalePrice')

# %%
#creating a pivot table of the "SalePrice" and "OverallQual" 
#and for generating a bar chart that shows the median sale price for each overall quality level.
#  
saleprice_overall_quality = train_df.pivot_table(index='OverallQual', values='SalePrice',aggfunc=np.median)
saleprice_overall_quality.plot(kind='bar',color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.show()

# %%
saleprice_garage_area = train_df.pivot_table(index='GarageArea', values='SalePrice',aggfunc=np.median)
saleprice_overall_quality.plot(kind='bar',color='yellow')
plt.xlabel('GarageArea')
plt.ylabel('Median Sale Price')
plt.show()

# %%
var = 'OverallQual'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis= 1)
f, ax = plt.subplots(figsize=(12,8))
fig = sns.boxplot(x=var,y='SalePrice',data=data)
fig.axis(ymin=0, ymax =800000)

# %%
var = 'Neighborhood'
data = pd.concat([train_df['SalePrice'], train_df[var]],axis=1)
f, ax = plt.subplots(figsize=(16,10))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax =800000)
xt = plt.xticks(rotation=45)

# %%
plt.figure(figsize=(12,6))
sns.countplot(x='Neighborhood', data= data)
xt = plt.xticks(rotation=45)

# %%
for c in categorical_features:
    train_df[c] = train_df[c].astype('category')
    if train_df[c].isnull().any():
        train_df[c] = train_df[c].cat.add_categories(['MISSING'])
        train_df[c] = train_df[c].fillna('MISSING')


def boxplot(x,y, **kwargs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation=90)

f = pd.melt(train_df, id_vars=['SalePrice'], value_vars=categorical_features)
g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False,sharey=False,height=5)
g = g.map(boxplot, "value","SalePrice")

# %%
var = 'SaleType'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(16,10))
fig = sns.boxplot(x=var,y='SalePrice', data=data)
fig.axis(ymin=0, ymax = 800000)
xt = plt.xticks(rotation=45)

# %%
var = 'SaleCondition'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
f, ax = plt.subplots(figsize=(16,10))
fig = sns.boxplot(x=var,y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
xt = plt.xticks(rotation=45)

# %%
# visualizing the distribution of the "SalePrice" data for each level of the "Functional" variable
sns.violinplot(x = 'Functional',y='SalePrice', data=train_df)


# %%
sns.catplot(
    x='FireplaceQu',
    y='SalePrice',
    data = train_df,
    color ='m',
    estimator = np.median,
    order = ['Ex','Gd', 'TA' , 'Fa' , 'Po'],
    height = 4.5,
    aspect = 1.35
    )

# %%
g = sns.FacetGrid(train_df, col='FireplaceQu', col_wrap=3, col_order=['Ex', 'Gd', 'TA', 'Fa', 'Po'])
g.map(sns.boxplot, 'Fireplaces', 'SalePrice', order = [1, 2 , 3], palette = 'Set2')

# %%
plt.figure(figsize=(8,10))
g1 = sns.pointplot(x='Neighborhood', y='SalePrice',data=train_df, hue='LotShape')
g1.set_xticklabels(g1.get_xticklabels(), rotation = 90)
g1.set_title('LotShape Based on Neighborhood', fontsize = 15)
g1.set_xlabel('Neighborhood')
g1.set_ylabel('Sale Price', fontsize =12)
plt.show()

# %%
#Missing Value Analysis for numeric types
total = numeric_features.isnull().sum().sort_values(ascending=False)
percent = (numeric_features.isnull().sum() / numeric_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, join='outer',keys=['Total Missing Count', '% of Total Observations'])
missing_data.index_name = 'Numeric Feature'
missing_data.head(20)

# %%
#bar chat rep of missing numeric values
missing_values = numeric_features.isnull().sum(axis=0).reset_index()
missing_values.columns = ['column_name', 'missing_count']
missing_values = missing_values.loc[missing_values['missing_count']>0]
missing_values = missing_values.sort_values(by='missing_count')

ind = np.arange(missing_values.shape[0])
width = 0.1
#only 3 missing values in numeric df
fig, ax = plt.subplots(figsize=(12,3))
rects = ax.barh(ind,missing_values.missing_count.values, color = 'b')
ax.set_yticks(ind)
ax.set_yticklabels(missing_values.column_name.values, rotation = 'horizontal')
ax.set_xlabel('Missing Observations Count')
ax.set_title('Missing Observations Count - Numeric Features')
plt.show()

# %%
#Missing Value Analysis for categorical types
total = categorical_features.isnull().sum().sort_values(ascending=False)
percent = (categorical_features.isnull().sum() / categorical_features.isnull().count()).sort_values(ascending =False)
missing_data = pd.concat([total, percent], axis =1, join = 'outer', keys = ['Total Missing Count', '% of Total Observations'])
missing_data.index.name = 'Feature'
missing_data.head(20)

# %%
#bar rep of categorical missing values
missing_values = categorical_features.isnull().sum(axis=0).reset_index()
missing_values.columns = ['column_name', 'missing_count']
missing_values = missing_values.loc[missing_values['missing_count']>0]
missing_values = missing_values.sort_values(by='missing_count')

ind = np.arange(missing_values.shape[0])
width = 0.5
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind,missing_values.missing_count.values, color = 'r')
ax.set_yticks(ind)
ax.set_yticklabels(missing_values.column_name.values, rotation = 'horizontal')
ax.set_xlabel('Missing Observations Count')
ax.set_title('Missing Observations Count - Categorical Features')
plt.show()

# %%
for column_name in train_df.columns:
    if train_df[column_name].dtypes == 'object':
        train_df[column_name] = train_df[column_name].fillna(train_df[column_name].mode().iloc[0])
        unique_category = len(train_df[column_name].unique())
        print('Feature in train set {} has {} unique categories'.format(column_name, unique_category))


# %%
#both(test/train) are almost similar
for column_name in test_df.columns:
    if test_df[column_name].dtypes == 'object':
        train_df[column_name] = test_df[column_name].fillna(test_df[column_name].mode().iloc[0])
        unique_category = len(test_df[column_name].unique())
        print('Feature in test set {} has {} unique categories'.format(column_name,unique_category))


