import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# Load the dataset from the CSV file
data = pd.read_csv('C:/Users/speed/Desktop/python licenta/aqi2.csv')

# Split the dataset into features (X) and target (y)
X = data.drop(['Date', 'Location', 'AQI'], axis=1)
y = data['AQI']

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the regression algorithms
algorithms = {
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'K Neighbors': KNeighborsRegressor(),
    'Support Vector Regression': LinearSVR(max_iter=10000),
    
}

# Define the evaluation metrics
metrics = {
    'Explained Variance R2 Score': make_scorer(explained_variance_score),
    'Mean Absolute Error': make_scorer(mean_absolute_error)
}

# Initialize the results dictionary
results = {metric: {name: [] for name in algorithms.keys()} for metric in metrics.keys()}

# Evaluarea fiecarui algorithm folosind metoda validare încrucișată
def evaluate_algorithm(name, model, metric_scorer):
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring=metric_scorer)
    return scores

for name, model in algorithms.items():
    for metric_name, metric_scorer in metrics.items():
        # Perform parallel evaluation and calculate the metric scores
        scores = Parallel(n_jobs=-1)(delayed(evaluate_algorithm)(name, model, metric_scorer) for _ in range(5))
        metric_scores = [score.mean() for score in scores]
        results[metric_name][name] = metric_scores

# Print the results
for metric, metric_results in results.items():
    print(f"--- {metric} ---")
    for name, scores in metric_results.items():
        print(f"{name}: {sum(scores) / len(scores):.4f} (+/- {2 * (max(scores) - min(scores)):.4f})")

    best_algorithm = min(metric_results, key=lambda x: sum(metric_results[x]) / len(metric_results[x])) if metric == 'Mean Absolute Error' else max(metric_results, key=lambda x: sum(metric_results[x]) / len(metric_results[x]))
    improvement = sum(metric_results[best_algorithm]) / len(metric_results[best_algorithm])

    if metric == 'Explained Variance R2 Score':
        explanation = "higher values indicate a better fit to the data."
    elif metric == 'Mean Absolute Error':
        explanation = "lower values indicate better accuracy in predicting AQI."

    print(f"\nBest Algorithm: {best_algorithm}")
    print(f"Reason: The {best_algorithm} algorithm achieved the best {metric} of {improvement:.4f}.")
    print(f"This means that the {best_algorithm} algorithm provides {explanation}")
    print("--------------------------------------")
