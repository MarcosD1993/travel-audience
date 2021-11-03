# Senior Data Analyst - Technical Exercise

# Import packages
#region

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy import stats

#endregion

# Import data
print('Please make sure your data set is stored in the same directory as this script!')
df = pd.read_excel('Interview_HW.xlsx')

# Data Quality
#region

print('Data Quality:')

## Get overview
display(df)
display(list(df))
display(df.shape)
display(df.describe().round(2))

## Data types
display(df.dtypes)
df_cleaned = df.convert_dtypes()
display(df_cleaned.dtypes)
df_cleaned['search_ts'] = pd.to_datetime(df_cleaned['search_ts'])
display(df_cleaned.dtypes)

## NAs
display(df_cleaned.isna().sum())
df_cleaned = df_cleaned.dropna()
print('Removed NAs')
display(df_cleaned.shape)

## Duplicates
list_columns = list(df)
df_deduped = df_cleaned.drop_duplicates()
print('Removed duplicates (all columns)')
display(df_deduped.shape)

list_columns.remove('user_ID')
df_deduped = df_deduped.drop_duplicates(list_columns)
print('Removed duplicates (all columns except ID)')
display(df_deduped.shape)

list_columns.remove('search_ts')
#df_deduped_strict = df_deduped.drop_duplicates(list_columns)
#print('Removed duplicates (all columns except ID and time stamp)')
#display(df_deduped_strict.shape)

## Outliers
### Get overview
list_columns_outliers = df.select_dtypes(include=np.number).columns.tolist()

for column_outliers in list_columns_outliers:
    plt.figure()
    sns.boxplot(data=df[column_outliers]).set_title(column_outliers + ' (raw)')

### Set maximum number of days
max_value = 365 # in days

### Days to departure
#### Remove negative values
df_no_outliers = df_deduped[df_deduped['days_to_departure'] >= 0]
print('Days to departure: removed negative values')
display(df_no_outliers.shape)

#### Remove unrealistic values
# df_no_outliers = df_no_outliers[df_no_outliers['days_to_departure'] <= max_value]
# print('Days to departure: removed unrealistic values')
# display(df_no_outliers.shape)

### Trip duration
#### Remove negative values
df_no_outliers = df_no_outliers[df_no_outliers['trip_duration'] >= 0]
print('Trip duration: removed negative values')
display(df_no_outliers.shape)

#### Remove unrealistic values
df_no_outliers = df_no_outliers[df_no_outliers['trip_duration'] <= max_value]
print('Trip duration: removed unrealistic values')
display(df_no_outliers.shape)

### Distance
#### Remove unrealistic values
earth_perimeter = 40000 # in kilometers
df_no_outliers['distance'] = df_no_outliers['distance'] / 1000

df_no_outliers = df_no_outliers[df_no_outliers['distance'] <= earth_perimeter]
print('Distance: removed unrealistic values')
display(df_no_outliers.shape)

### Get cleaned overview
display(df_no_outliers.describe().round(0))

for column_outliers in list_columns_outliers:
    plt.figure()
    sns.boxplot(data=df_no_outliers[column_outliers].values).set_title(column_outliers + ' (cleaned)')

#endregion

# Trends
#region

print('Trends:')

## Prepare data
df_trends = df_no_outliers

df_trends['search_ts'].describe(datetime_is_numeric=True)
df_trends['month'] = pd.DatetimeIndex(df_trends['search_ts']).month
df_trends['day'] = pd.DatetimeIndex(df_trends['search_ts']).day

criteria = [df_trends['day'].between(1, 10), df_trends['day'].between(11, 20), df_trends['day'].between(21, 31)]
categories = ['beginning', 'mid', 'end']

df_trends['day_cat'] = np.select(criteria, categories, 0)

df_trends['weekday'] = df_trends['search_ts'].dt.day_name()

list_alpha_2 = [i.alpha_2 for i in list(pycountry.countries)]
list_name = [i.name for i in list(pycountry.countries)]
dict_countries = dict(zip(list_alpha_2, list_name))
df_trends['user_country_source1'] = df_trends['user_country_source1'].map(dict_countries)
df_trends['user_country_source2'] = df_trends['user_country_source2'].map(dict_countries)

# Get top 5
display(df_trends['user_city_source1'].value_counts().nlargest(5))
display(df_trends['user_country_source1'].value_counts().nlargest(5))
display(df_trends['user_city_source2'].value_counts().nlargest(5))
display(df_trends['user_country_source2'].value_counts().nlargest(5))
display(df_trends['searched_destination'].value_counts().nlargest(5))

# Group by
display(df_trends.groupby(by=['month']).mean())
display(df_trends.groupby(by=['day_cat']).mean())
display(df_trends.groupby(by=['weekday']).mean())

display(df_trends.groupby(by=['month'])['user_city_source1', 'user_country_source1', 'user_city_source2', 'user_country_source2', 'searched_destination'].agg(pd.Series.mode))
display(df_trends.groupby(by=['day_cat'])['user_city_source1', 'user_country_source1', 'user_city_source2', 'user_country_source2', 'searched_destination'].agg(pd.Series.mode))
display(df_trends.groupby(by=['weekday'])['user_city_source1', 'user_country_source1', 'user_city_source2', 'user_country_source2', 'searched_destination'].agg(pd.Series.mode))

#endregion

# Correlations
#region

print('Correlations:')

## Encode categorical columns
df_trends_cat = df_trends.copy()

list_columns_categorical = ['user_city_source1', 'user_country_source1', 'user_city_source2', 'user_country_source2', 'website_language', 'searched_destination', 'day_cat', 'weekday']

for column_categorical in list_columns_categorical:
    df_trends_cat[column_categorical] = df_trends_cat[column_categorical].astype('category').cat.codes

## Set variables
list_variables_independent = ['user_city_source1', 'user_city_country1', 'user_city_source2', 'user_city_country2', 'website_language', 'month', 'day', 'day_cat', 'weekday']
list_variables_dependent = ['days_to_departure', 'trip_duration', 'searched_destination', 'distance']

## Check for normal distribution
for variable in list_variables_dependent:
    print(variable)

    k2, p = stats.normaltest(df_trends_cat[variable])
    alpha = 1e-3
    print("p = {:g}".format(p))

    # Null hypothesis: data is normally distributed
    if p < alpha:
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")
    print()

    plt.figure()
    sns.histplot(data=df_trends_cat[variable]).set_title(variable)

## Spearman
def correlation_spearman(df):
    r = df.corr(method="spearman").round(2)
    plt.figure(figsize=(10,6))
    heatmap = sns.heatmap(r, vmin=-1, vmax=1, annot=True)
    plt.title("Spearman Correlation")

    return(r)

correlation_spearman(df_trends_cat)

#endregion