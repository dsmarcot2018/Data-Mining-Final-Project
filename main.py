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

corr_matrix = data.corr()

# X_default = data.drop(['degrees_from_mean'], axis=1) # -0.00162
# X_no_id = data.drop(['id', 'degrees_from_mean'], axis=1) # -0.00143
# X_no_date = data.drop(['date_str', 'degrees_from_mean'], axis=1) # -0.00153
# X_no_long = data.drop(['longitude', 'degrees_from_mean'], axis=1) # -0.00234
# X_no_lat = data.drop(['latitude', 'degrees_from_mean'], axis=1) # -0.00277
# X_no_max = data.drop(['max_temp', 'degrees_from_mean'], axis=1) # 0.00074
# X_no_min = data.drop(['min_temp', 'degrees_from_mean'], axis=1) # 0.00031
# X_no_type = data.drop(['type', 'degrees_from_mean'], axis=1) # -0.00287
# X_no_id_date_max_min = data.drop(['id', 'date_str', 'max_temp', 'min_temp', 'degrees_from_mean'], axis=1) # 0.00032
# X_no_id_date_max = data.drop(['id', 'date_str', 'max_temp', 'degrees_from_mean'], axis=1) # 0.00057
# X_no_id_date_min = data.drop(['id', 'date_str', 'min_temp', 'degrees_from_mean'], axis=1) # -0.00029
# X_no_id_max_min = data.drop(['id', 'max_temp', 'min_temp', 'degrees_from_mean'], axis=1) # 0.00017
# X_no_date_max_min = data.drop(['date_str', 'max_temp', 'min_temp', 'degrees_from_mean'], axis=1) # -0.00005
# X_no_id_date = data.drop(['id', 'date_str', 'degrees_from_mean'], axis=1) # -0.00168
# X_no_id_max = data.drop(['id', 'max_temp', 'degrees_from_mean'], axis=1) # 0.00074
X_no_id_min = data.drop(['id', 'min_temp', 'degrees_from_mean'], axis=1) # 0.00012
# X_no_date_max = data.drop(['date_str', 'max_temp', 'degrees_from_mean'], axis=1) # -0.00017
# X_no_date_min = data.drop(['date_str', 'min_temp', 'degrees_from_mean'], axis=1) # -0.00066
# X_no_max_min = data.drop(['max_temp', 'min_temp', 'degrees_from_mean'], axis=1) # 0.00003

def run(X):
    y = data['degrees_from_mean']
    
    regressor = RandomForestRegressor(verbose=3, n_estimators=10, n_jobs=-1)
    regressor.fit(X, y)

    y_pred = regressor.predict(X)

    y_compare = []

    for pred, actual in zip(y_pred, y):
        y_compare.append(pred - actual)

    y_compare_mean = round((sum(y_compare) / len(y_compare)), 5)

    print(y_compare_mean)
    return y_compare_mean

# X_no_min_accuracy = []
# X_no_id_date_max_min_accuracy = []
# X_no_id_date_min_accuracy = []
# X_no_id_max_min_accuracy = []
# X_no_date_max_min_accuracy = []
X_no_id_min_accuracy = []
# X_no_date_max_accuracy = []
# X_no_max_min_accuracy = []

for i in range(10):
    # run(X_default)
    # run(X_no_id)
    # run(X_no_date)
    # run(X_no_long)
    # run(X_no_lat)
    # run(X_no_max)
    # X_no_min_accuracy.append(run(X_no_min))
    # run(X_no_type)
    # X_no_id_date_max_min_accuracy.append(run(X_no_id_date_max_min))
    # run(X_no_id_date_max)
    # X_no_id_date_min_accuracy.append(run(X_no_id_date_min))
    # X_no_id_max_min_accuracy.append(run(X_no_id_max_min))
    # X_no_date_max_min_accuracy.append(run(X_no_date_max_min))
    # run(X_no_id_date)
    # run(X_no_id_max)
    X_no_id_min_accuracy.append(run(X_no_id_min))
    # X_no_date_max_accuracy.append(run(X_no_date_max))
    # run(X_no_date_min)
    # X_no_max_min_accuracy.append(run(X_no_max_min))

# print('X_no_min_accuracy: ', round((sum(X_no_min_accuracy) / len(X_no_min_accuracy)), 5))
# print('X_no_id_date_max_min_accuracy: ', round((sum(X_no_id_date_max_min_accuracy) / len(X_no_id_date_max_min_accuracy)), 5))
# print('X_no_id_date_min_accuracy: ', round((sum(X_no_id_date_min_accuracy) / len(X_no_id_date_min_accuracy)), 5))
# print('X_no_id_max_min_accuracy: ', round((sum(X_no_id_max_min_accuracy) / len(X_no_id_max_min_accuracy)), 5))
# print('X_no_date_max_min_accuracy: ', round((sum(X_no_date_max_min_accuracy) / len(X_no_date_max_min_accuracy)), 5))
print('X_no_id_min_accuracy: ', round((sum(X_no_id_min_accuracy) / len(X_no_id_min_accuracy)), 5))
# print('X_no_date_max_accuracy: ', round((sum(X_no_date_max_accuracy) / len(X_no_date_max_accuracy)), 5))
# print('X_no_max_min_accuracy: ', round((sum(X_no_max_min_accuracy) / len(X_no_max_min_accuracy)), 5))

print(corr_matrix)
print(data.head())
print(data.info())
