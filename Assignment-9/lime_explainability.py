
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

# Create output directory
os.makedirs("results", exist_ok=True)

# Load the trained model and test data
model = joblib.load("results/random_forest_model.pkl")
X_test = pd.read_csv("results/X_test.csv")

# Class names corresponding to model classes 0 and 1
class_names = ["malignant", "benign"]

# Prediction wrapper: preserve feature names for scikit-learn
def predict_fn(data):
    data = pd.DataFrame(data, columns=X_test.columns)
    return model.predict_proba(data)

# Create the LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_test.to_numpy(),
    feature_names=X_test.columns.tolist(),
    class_names=class_names,
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

# Select one test instance
instance = X_test.iloc[0].to_numpy()
instance_df = X_test.iloc[[0]]

# Get model prediction and probabilities
prediction = int(model.predict(instance_df)[0])
probabilities = model.predict_proba(instance_df)[0]

# Generate explanation for the predicted class
explanation = explainer.explain_instance(
    data_row=instance,
    predict_fn=predict_fn,
    labels=(prediction,),
    num_features=10,
    num_samples=3000
)

# Display results
print("LIME Explainability Results")
print("-" * 40)
print(f"Predicted class: {class_names[prediction]}")
print(f"Malignant probability: {probabilities[0]:.4f}")
print(f"Benign probability: {probabilities[1]:.4f}")
print("\nTop feature contributions:")

for feature, contribution in explanation.as_list(label=prediction):
    print(f"{feature}: {contribution:.4f}")

# Save the explanation as an HTML report
explanation.save_to_file("results/lime_explanation.html")

# Generate and save the feature contribution plot
fig = explanation.as_pyplot_figure(label=prediction)
fig.tight_layout()
fig.savefig(
    "results/lime_explanation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("\nLIME explainability completed successfully!")
print("Generated files:")
print("- results/lime_explanation.html")
print("- results/lime_explanation.png")
