import pandas as pd
import numpy as np

df = pd.read_csv(r'C:/Users/PC\Desktop/KIDOLA/Kinonapython/train.csv')
print(df,"  ")

vab1 = pd.read_csv('C:/Users/PC\Desktop/KIDOLA/Kinonapython/train.csv')
vab11 = pd.read_csv('C:/Users/PC\Desktop/KIDOLA/Kinonapython/train.csv') 
vab22 = pd.read_csv('C:/Users/PC/Desktop/KIDOLA/Kinonapython/test.csv')  
print(vab1.describe(),"  ")
print("Don't get discouraged, keep trying","  ")

vab1.head()
print(vab1.head(),"  ")

vab2 = pd.read_csv('C:/Users/PC\Desktop/KIDOLA/Kinonapython/train.csv')
vab2.tail()
print(vab2.tail(),"  ")

vab1.shape
vab2.shape
print(vab1.shape,vab2.shape,"  ")


numr_charact = vab1.select_dtypes(include=[np.number])
numr_charact.columns
print(numr_charact.columns," ")

listings_charact = vab1.select_dtypes(include=[np.object])
listings_charact.columns
print(listings_charact.columns," ")

import missingno as msno
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st



print(vab1.info(),"  ")

vab1.isna().sum()
print(df.isna().sum(),"  ")

vab3 = msno.matrix(vab1.sample(500))
print(vab3,"  ")

msno.heatmap(vab1)
print(msno.heatmap(vab1),"  ")

print(msno.heatmap(vab1),"    ")
print(msno.bar(vab1.sample(1000)),"    ")

print(msno.dendrogram(vab1),"  ")

print(vab1.skew()," ")

print(vab1.kurt(), "  ")
y = vab1['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)

sns.distplot(vab1.skew(),color='blue',axlabel ='Skewness')

plt.figure(figsize = (12,8))
sns.distplot(vab1.kurt(),color='r',axlabel ='Kurtosis',norm_hist= False, kde = True,rug = False)
#plt.hist(train.kurt(),orientation = 'vertical',histtype = 'bar',label ='Kurtosis', color ='blue')
plt.show()

plt.hist(vab1['SalePrice'],orientation = 'vertical',histtype = 'bar', color ='blue')
plt.show()

interconnections = numr_charact.corr()
print(interconnections['SalePrice'].sort_values(ascending = False),'\n')

f , ax = plt.subplots(figsize = (14,12))
plt.title('Correlations between the numeric variables and the sale price ',y=1,size=16)
sns.heatmap(interconnections,square = True,  vmax=0.8)

sns.set()
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
sns.pairplot(vab1[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()

saleprice_overall_quality= vab1.pivot_table(index ='OverallQual',values = 'SalePrice', aggfunc = np.median)
saleprice_overall_quality.plot(kind = 'bar',color = 'blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.show()

numr_charact = vab1.select_dtypes(include=[np.number])
interconnections = numr_charact.corr()
k= 11

cols = interconnections.nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(vab1[cols].values.T)
f , ax = plt.subplots(figsize = (14,12))
sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis')

sns.set()
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
sns.pairplot(vab11[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()

fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(14,10))
OverallQual_scatter_plot = pd.concat([vab11['SalePrice'],vab11['OverallQual']],axis = 1)
sns.regplot(x='OverallQual',y = 'SalePrice',data = OverallQual_scatter_plot,scatter= True, fit_reg=True, ax=ax1)
TotalBsmtSF_scatter_plot = pd.concat([vab11['SalePrice'],vab11['TotalBsmtSF']],axis = 1)
sns.regplot(x='TotalBsmtSF',y = 'SalePrice',data = TotalBsmtSF_scatter_plot,scatter= True, fit_reg=True, ax=ax2)
GrLivArea_scatter_plot = pd.concat([vab11['SalePrice'],vab11['GrLivArea']],axis = 1)
sns.regplot(x='GrLivArea',y = 'SalePrice',data = GrLivArea_scatter_plot,scatter= True, fit_reg=True, ax=ax3)
GarageArea_scatter_plot = pd.concat([vab11['SalePrice'],vab11['GarageArea']],axis = 1)
sns.regplot(x='GarageArea',y = 'SalePrice',data = GarageArea_scatter_plot,scatter= True, fit_reg=True, ax=ax4)
FullBath_scatter_plot = pd.concat([vab11['SalePrice'],vab11['FullBath']],axis = 1)
sns.regplot(x='FullBath',y = 'SalePrice',data = FullBath_scatter_plot,scatter= True, fit_reg=True, ax=ax5)
YearBuilt_scatter_plot = pd.concat([vab11['SalePrice'],vab11['YearBuilt']],axis = 1)
sns.regplot(x='YearBuilt',y = 'SalePrice',data = YearBuilt_scatter_plot,scatter= True, fit_reg=True, ax=ax6)
YearRemodAdd_scatter_plot = pd.concat([vab11['SalePrice'],vab11['YearRemodAdd']],axis = 1)
YearRemodAdd_scatter_plot.plot.scatter('YearRemodAdd','SalePrice')

var = 'OverallQual'
data = pd.concat([vab11['SalePrice'], vab11[var]], axis=1)
f, ax = plt.subplots(figsize=(12, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

var = 'Neighborhood'
data = pd.concat([vab1['SalePrice'], vab1[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
xt = plt.xticks(rotation=45)

plt.figure(figsize = (12, 6))
sns.countplot(x = 'Neighborhood', data = data)
xt = plt.xticks(rotation=45)

for c in listings_charact:
    vab1[c] = vab1[c].astype('category')
    if vab1[c].isnull().any():
        vab1[c] = vab1[c].cat.add_categories(['MISSING'])
        vab1[c] = vab1[c].fillna('MISSING')

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)
f = pd.melt(vab1, id_vars=['SalePrice'], value_vars=listings_charact)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(boxplot, "value", "SalePrice")

var = 'SaleType'
data = pd.concat([vab1['SalePrice'], vab1[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
xt = plt.xticks(rotation=45)

var = 'SaleCondition'
data = pd.concat([vab1['SalePrice'], vab1[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
xt = plt.xticks(rotation=45)

g = sns.FacetGrid(vab1, col = 'FireplaceQu', col_wrap = 3, col_order=['Ex', 'Gd', 'TA', 'Fa', 'Po'])
g.map(sns.boxplot, 'Fireplaces', 'SalePrice', order = [1, 2, 3], palette = 'Set2')

plt.figure(figsize=(8,10))
g1 = sns.pointplot(x='Neighborhood', y='SalePrice', 
                   data=vab1, hue='LotShape')
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
g1.set_title("Lotshape Based on Neighborhood", fontsize=15)
g1.set_xlabel("Neighborhood")
g1.set_ylabel("Sale Price", fontsize=12)
plt.show()

total = numr_charact.isnull().sum().sort_values(ascending=False)
percent = (numr_charact.isnull().sum()/numr_charact.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['Total Missing Count', '% of Total Observations'])
missing_data.index.name =' Numeric Feature'
missing_data.head(20)

missing_values = numr_charact.isnull().sum(axis=0).reset_index()
missing_values.columns = ['column_name', 'missing_count']
missing_values = missing_values.loc[missing_values['missing_count']>0]
missing_values = missing_values.sort_values(by='missing_count')

ind = np.arange(missing_values.shape[0])
width = 0.1
fig, ax = plt.subplots(figsize=(12,3))
rects = ax.barh(ind, missing_values.missing_count.values, color='b')
ax.set_yticks(ind)
ax.set_yticklabels(missing_values.column_name.values, rotation='horizontal')
ax.set_xlabel("Missing Observations Count")
ax.set_title("Missing Observations Count - Numeric Features")
plt.show()

total = listings_charact.isnull().sum().sort_values(ascending=False)
percent = (listings_charact.isnull().sum()/listings_charact.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['Total Missing Count', ' % of Total Observations'])
missing_data.index.name ='Feature'
missing_data.head(20)

missing_values = listings_charact.isnull().sum(axis=0).reset_index()
missing_values.columns = ['column_name', 'missing_count']
missing_values = missing_values.loc[missing_values['missing_count']>0]
missing_values = missing_values.sort_values(by='missing_count')

ind = np.arange(missing_values.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_values.missing_count.values, color='red')
ax.set_yticks(ind)
ax.set_yticklabels(missing_values.column_name.values, rotation='horizontal')
ax.set_xlabel("Missing Observations Count")
ax.set_title("Missing Observations Count - Categorical Features")
plt.show()

for column_name in vab1.columns:
    if vab1[column_name].dtypes == 'object':
        vab1[column_name] = vab1[column_name].fillna(vab1[column_name].mode().iloc[0])
        unique_category = len(vab1[column_name].unique())
        print("Feature '{column_name}' has '{unique_category}' unique categories".format(column_name = column_name,
                                                                                         unique_category=unique_category))
 
for column_name in vab22.columns:
    if vab22[column_name].dtypes == 'object':
        vab22[column_name] = vab22[column_name].fillna(vab22[column_name].mode().iloc[0])
        unique_category = len(vab22[column_name].unique())
        print("Features in test set '{column_name}' has '{unique_category}' unique categories".format(column_name = column_name, unique_category=unique_category))


