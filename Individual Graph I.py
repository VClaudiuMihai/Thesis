import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import learning_curve

# Load the dataset from the CSV file
data = pd.read_csv('C:/Users/speed/Desktop/python licenta/aqi2.csv')

# Split the dataset into features (X) and target (y)
X = data.drop(['Date', 'Location', 'AQI'], axis=1)
y = data['AQI']

# Initialize the regression algorithms
algorithms = {
    'Random Forest': RandomForestRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'K Neighbors': KNeighborsRegressor(),
    'SVR': SVR()
}

# Define the evaluation metric
metric = make_scorer(mean_absolute_error)

# Plot the learning curves for each algorithm
for name, model in algorithms.items():
    # Calculate the learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5,
        scoring=metric, shuffle=True, random_state=42)

    # Calculate the mean and standard deviation of the train and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Error', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.3)
    plt.plot(train_sizes, test_mean, label='Validation Error', marker='o')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.3)
    plt.xlabel('Training Examples')
    plt.ylabel('Mean Absolute Error')
    plt.title(f'Learning Curve - {name}')
    plt.legend()
    plt.grid(True)
    plt.show()
