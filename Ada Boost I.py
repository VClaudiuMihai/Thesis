import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
from sklearn.model_selection import train_test_split

# Încărcare set de date din fișierul CSVe
data = pd.read_csv('C:/Users/speed/Desktop/python licenta/aqi2.csv')

# Impărțire set de date în caracteristici (X) și țintă (Y)
X = data.drop(['Date', 'Location', 'AQI'], axis=1)
y = data['AQI']

# Împărțire a datelor în seturi de instruire și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creați modelul de regresie Ada Boost
model = AdaBoostRegressor(random_state=42)

# Potriviți modelul la datele de antrenament
model.fit(X_train, y_train)

# Prezicere AQI pentru datele de testare
y_pred = model.predict(X_test)

# Calcularea valorilor de evaluare
explained_variance = explained_variance_score(y_test, y_pred)
mean_absolute_err = mean_absolute_error(y_test, y_pred)
mean_absolute_percentage_err = mean_absolute_percentage_error(y_test, y_pred)
median_absolute_err = median_absolute_error(y_test, y_pred)

# Tipărirea valorilor de evaluare
print("Evaluation Metrics:")
print("Explained Variance R2 Score:", explained_variance)
print("Mean Absolute Error:", mean_absolute_err)
print("Mean Absolute Percentage Error:", mean_absolute_percentage_err)
print("Median Absolute Error:", median_absolute_err)

# Prezicerea valorilor AQI pentru ziua următoare în fiecare locație
locations = data['Location'].unique()  # Obțineți locații unice
next_data = pd.DataFrame(columns=['Location', 'AQI'])  # DataFrame pentru a stoca valorile AQI estimate

for location in locations:
    next_day_data = data[data['Location'] == location].tail(1).drop(['Date', 'Location', 'AQI'], axis=1)
    next_day_aqi = model.predict(next_day_data)
    next_data = pd.concat([next_data, pd.DataFrame({'Location': [location], 'AQI': next_day_aqi})], ignore_index=True)

# Tipărirea valorilor AQI estimate pentru ziua următoare în fiecare locație
print("Valorile AQI estimate pentru ziua urmatoare:")
print(next_data)
