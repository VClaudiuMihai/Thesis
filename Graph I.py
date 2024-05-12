import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, make_scorer
from sklearn.model_selection import cross_validate

# Load the dataset from the CSV file
data = pd.read_csv('C:/Users/speed/Desktop/python licenta/aqi2.csv')

# Split the dataset into features (X) and target (y)
X = data.drop(['Date', 'Location', 'AQI'], axis=1)
y = data['AQI']

# Initialize the regression algorithms
algorithms = {
    'Random Forest': RandomForestRegressor(),
    'AdaBoost': AdaBoostRegressor(base_estimator=DecisionTreeRegressor()),
    'Gradient Boosting': GradientBoostingRegressor(),
    'K Neighbors': KNeighborsRegressor(),
    'Support Vector Regression': SVR(),
}

# Define the evaluation metrics
metrics = {
    'Explained Variance R2 Score': make_scorer(explained_variance_score),
    'Mean Absolute Error': make_scorer(mean_absolute_error)
}

# Initialize the results dictionary
results = {metric: {name: [] for name in algorithms.keys()} for metric in metrics.keys()}

# Evaluate each algorithm using cross-validation
for name, model in algorithms.items():
    for metric_name, metric_scorer in metrics.items():
        # Perform cross-validation and calculate the evaluation metric scores
        cv_results = cross_validate(model, X, y, cv=5, scoring=metric_scorer)
        metric_scores = cv_results['test_score']
        results[metric_name][name] = metric_scores

# Plot and save the results
for metric, metric_results in results.items():
    names = list(metric_results.keys())
    scores = [scores.mean() for scores in metric_results.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(names, scores)
    plt.xlabel('Algorithm')
    plt.ylabel(metric)
    plt.title(f'Performanța algoritmilor - {metric}')
    plt.savefig(f'{metric}.png')
    plt.show()
