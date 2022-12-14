from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import accuracy_score

# Read the data from the CSV file and drop the 'serialid' and 'station_name' columns
data = pd.read_csv('weather-anomalies-1964-2013.csv')
data = data.drop(['serialid', 'station_name'], axis=1)

# Drop the unnecessary values from id and date_str columns and change id to int
data['id'] = [int(id[3:]) for id in data['id']]
data['date_str'] = [date[5:] for date in data['date_str']]
data['date_str'] = [date[:2] + date[3:] for date in data['date_str']]
data['date_str'] = [int(date) for date in data['date_str']]

# Replace the string values in the 'type' column with numerical values
data['type'].replace('Weak Hot', 0, inplace=True)
data['type'].replace('Strong Hot', 1, inplace=True)
data['type'].replace('Weak Cold', 2, inplace=True)
data['type'].replace('Strong Cold', 3, inplace=True)

# Separate the features and the target variable
X = data.drop(['degrees_from_mean'], axis=1)
y = data['degrees_from_mean']

# Create correlation matrix
corr_matrix = data.corr()

# Create a random forest regressor and fit it to the data
regressor = RandomForestRegressor(verbose=3)
regressor.fit(X, y)

y_pred = regressor.predict(X)

y_compare = []

for pred, actual in zip(y_pred, y):
    y_compare.append(pred - actual)

y_compare_mean = round((sum(y_compare) / len(y_compare)), 5)
accuracy = accuracy_score(y, y_pred)

print(y_pred)
print(accuracy)
print(y_compare_mean)
print(corr_matrix)
print(data.head())
print(data.info())
