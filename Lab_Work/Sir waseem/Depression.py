import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Set random seed for reproducibility
np.random.seed(42)

# Generate an imbalanced dataset
X, y = make_classification(
    n_samples=4000,        # Total number of samples
    n_features=10,         # Number of features
    n_informative=6,       # Number of informative features
    n_redundant=2,         # Number of redundant features
    n_classes=2,           # Number of classes (depressed or not)
    weights=[0.8, 0.2],    # Imbalance ratio (80% not depressed, 20% depressed)
    random_state=42
)

# Create a DataFrame for the features
feature_columns = [f"feature_{i}" for i in range(1, 11)]
data = pd.DataFrame(X, columns=feature_columns)

# Add the target column
data['depression'] = y

# Save the dataset to a CSV file
data.to_csv("depression_dataset.csv", index=False)

print("Dataset created and saved as 'depression_dataset.csv'.")