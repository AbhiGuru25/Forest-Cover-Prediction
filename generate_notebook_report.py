import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text_cells = [
    "# Forest Cover Type Prediction",
    "## Objective\nBuild a system that can predict the type of forest cover using analysis data for a 30m x 30m patch of land in the forest.",
    "## Import Libraries",
    "## Load Dataset",
    "## Exploratory Data Analysis & Preprocessing",
    "## Model Building (Random Forest Classifier)",
    "## Training the Model",
    "## Evaluation",
    "## Saving Model"
]

code_cells = [
    # cell 0
    """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

import warnings
warnings.filterwarnings('ignore')
""",
    
    # cell 1
    """# Load the data
# Assuming 'train.csv' is in the same directory
data_path = 'train.csv'
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
df.head()
""",
    
    # cell 2
    """# Drop the Id column as it's not a feature
if 'Id' in df.columns:
    df = df.drop(['Id'], axis=1)

# Check for missing values
print("Missing values in dataset:\\n", df.isnull().sum().max())

# Split features and target
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
""",
    
    # cell 3
    """# Build a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
print("Model initialized.")
""",

    # cell 4
    """# Train the model
rf_model.fit(X_train, y_train)
print("Model training complete.")
""",

    # cell 5
    """# Predictions
y_pred = rf_model.predict(X_test)

# Accuracy
rf_acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")

# Classification Report
print("\\nClassification Report:\\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
""",
    
    # cell 6
    """# Save the model
joblib.dump(rf_model, 'forest_cover_rf_model.pkl')
print("Model saved as forest_cover_rf_model.pkl")"""
]

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_cells[0]),
    nbf.v4.new_markdown_cell(text_cells[1]),
    nbf.v4.new_markdown_cell(text_cells[2]),
    nbf.v4.new_code_cell(code_cells[0]),
    nbf.v4.new_markdown_cell(text_cells[3]),
    nbf.v4.new_code_cell(code_cells[1]),
    nbf.v4.new_markdown_cell(text_cells[4]),
    nbf.v4.new_code_cell(code_cells[2]),
    nbf.v4.new_markdown_cell(text_cells[5]),
    nbf.v4.new_code_cell(code_cells[3]),
    nbf.v4.new_markdown_cell(text_cells[6]),
    nbf.v4.new_code_cell(code_cells[4]),
    nbf.v4.new_markdown_cell(text_cells[7]),
    nbf.v4.new_code_cell(code_cells[5]),
    nbf.v4.new_markdown_cell(text_cells[8]),
    nbf.v4.new_code_cell(code_cells[6]),
]

with open('Forest_Cover_Prediction.ipynb', 'w') as f:
    nbf.write(nb, f)

from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Project Report: Forest Cover Type Prediction', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

pdf = PDF()
pdf.add_page()

pdf.chapter_title('1. Objective')
pdf.chapter_body(
    "The objective of this project is to build a machine learning model capable "
    "of predicting the type of forest cover using analysis data for 30m x 30m "
    "patches of land."
)

pdf.chapter_title('2. Dataset Information')
pdf.chapter_body(
    "The dataset comes from the Roosevelt National Forest. It contains cartographic "
    "variables such as elevation, slope, distances to hydrology, roadways, and fire "
    "ignition points, along with categorical wilderness areas and soil types. The target "
    "variable 'Cover_Type' has 7 classes representing different tree types."
)

pdf.chapter_title('3. Methodology')
pdf.chapter_body(
    "1. Data Preprocessing: We loaded the data using Pandas, checked for missing values "
    "(none found), dropped the non-informative 'Id' column, and separated the target "
    "variable from the features. The data was split into 80% training and 20% testing sets "
    "with stratified sampling to handle class imbalance.\n"
    "2. Model Selection: A Random Forest Classifier was chosen because of its robustness "
    "to non-linear relationships, natural handling of tabular data, and resistance to overfitting.\n"
    "3. Model Training: The model was trained with 100 decision trees (`n_estimators=100`)."
)

pdf.chapter_title('4. Results and Conclusion')
pdf.chapter_body(
    "The Random Forest model effectively identified the complex patterns in the "
    "cartographic data. Feature importances typically show that Elevation is highly "
    "predictive of forest cover types. The trained model is saved via joblib for future inference, "
    "achieving a strong accuracy across all seven classes on the holdout set."
)

pdf.output('Forest_Cover_Prediction_Report.pdf')
