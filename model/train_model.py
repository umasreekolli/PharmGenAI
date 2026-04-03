import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load
data = pd.read_csv('data/adr_dataset.csv')

# Check what's in Gender column
print("Gender values found:", data['Gender'].unique())

# Encode Gender safely (handles both text and number)
if data['Gender'].dtype == object:
    data['Gender'] = data['Gender'].map({'M': 0, 'F': 1})

# Encode CYP columns
cyp_map = {
    'Poor': 0, 'Intermediate': 1,
    'Normal': 2, 'Ultrarapid': 3,
    'Fast': 2, 'Rapid': 2
}
data['CYP2D6']  = data['CYP2D6'].map(cyp_map)
data['CYP3A4']  = data['CYP3A4'].map(cyp_map)
data['CYP2C19'] = data['CYP2C19'].map(cyp_map)

# Encode Drug
data['Drug'] = data['Drug'].astype('category').cat.codes

# Features & target
X = data.drop('ADR_Risk', axis=1)
y = data['ADR_Risk']

# Check for any remaining NaN
print("Any NaN values?", X.isnull().sum().sum())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
model = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    class_weight='balanced', random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(
    y_test, y_pred, target_names=['Low', 'Medium', 'High']
))

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/adr_model.pkl')
joblib.dump(list(X.columns), 'model/feature_names.pkl')
print("\nModel saved.")

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
joblib.dump(explainer, 'model/shap_explainer.pkl')

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('model/shap_importance.png')
plt.close()
print("Saved: shap_importance.png")

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('model/shap_summary.png')
plt.close()
print("Saved: shap_summary.png")

print("\nAll done!")
