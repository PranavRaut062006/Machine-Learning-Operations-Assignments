import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Dataset columns
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]



# Load dataset
data = pd.read_csv(
    "data/adult.csv",
    names=columns,
    skipinitialspace=True
)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)
print(data.head())


# Replace missing values
data = data.replace("?", pd.NA)

# Remove rows containing missing values
data = data.dropna()

print("Missing values handled!")
print("Dataset shape after cleaning:", data.shape)


# Separate features and target
X = data.drop("income", axis=1)
y = data["income"]

print("Features and target separated!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Define categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns

print("Categorical columns:", list(categorical_columns))
print("Numerical columns:", list(numerical_columns))


# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Data split completed!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Decision Tree model
decision_tree_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(
            random_state=42
        ))
    ]
)


# Random Forest model
random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# Train Decision Tree
print("\nTraining Decision Tree...")
decision_tree_pipeline.fit(X_train, y_train)

print("Decision Tree training completed!")

# Train Random Forest
print("\nTraining Random Forest...")
random_forest_pipeline.fit(X_train, y_train)

print("Random Forest training completed!")

# Make predictions
dt_predictions = decision_tree_pipeline.predict(X_test)
rf_predictions = random_forest_pipeline.predict(X_test)

# Calculate accuracy
dt_accuracy = accuracy_score(y_test, dt_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"Decision Tree Accuracy : {dt_accuracy:.4f}")
print(f"Random Forest Accuracy : {rf_accuracy:.4f}")
print("=" * 50)


# Save trained models
joblib.dump(
    decision_tree_pipeline,
    "models/decision_tree_model.pkl"
)

joblib.dump(
    random_forest_pipeline,
    "models/random_forest_model.pkl"
)

print("\nModels saved successfully!")