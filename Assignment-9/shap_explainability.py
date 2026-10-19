
import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("results", exist_ok=True)

# Load the trained model and test data
model = joblib.load("results/random_forest_model.pkl")
X_test = pd.read_csv("results/X_test.csv")

# Select a small sample for faster explanation
X_sample = X_test.iloc[:50]

# Create SHAP TreeExplainer for the Random Forest model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Handle SHAP output formats for binary classification
if isinstance(shap_values, list):
    # Older SHAP versions return a list, one array per class
    values_to_plot = shap_values[1]
else:
    # Newer SHAP versions may return a 3D array
    values_to_plot = (
        shap_values[:, :, 1]
        if shap_values.ndim == 3
        else shap_values
    )

# Generate global feature-importance summary
plt.figure()
shap.summary_plot(
    values_to_plot,
    X_sample,
    show=False,
    max_display=15
)
plt.tight_layout()
plt.savefig("results/shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# Generate a bar chart of mean absolute SHAP values
plt.figure()
shap.summary_plot(
    values_to_plot,
    X_sample,
    plot_type="bar",
    show=False,
    max_display=15
)
plt.tight_layout()
plt.savefig("results/shap_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

# Explain one individual prediction
single_instance = X_test.iloc[[0]]
single_shap_values = explainer.shap_values(single_instance)

if isinstance(single_shap_values, list):
    single_values = single_shap_values[1]
else:
    single_values = (
        single_shap_values[:, :, 1]
        if single_shap_values.ndim == 3
        else single_shap_values
    )

# Create a waterfall plot using the expected value and SHAP values
expected_value = explainer.expected_value
if hasattr(expected_value, "__len__"):
    expected_value = expected_value[1]

explanation = shap.Explanation(
    values=single_values[0],
    base_values=expected_value,
    data=single_instance.iloc[0].values,
    feature_names=X_test.columns.tolist()
)

plt.figure()
shap.plots.waterfall(explanation, max_display=15, show=False)
plt.savefig("results/shap_waterfall.png", dpi=300, bbox_inches="tight")
plt.close()

print("SHAP explainability completed successfully!")
print("Generated files:")
print("- results/shap_summary.png")
print("- results/shap_feature_importance.png")
print("- results/shap_waterfall.png")
