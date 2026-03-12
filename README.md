# Forest Cover Type Prediction 🌲

## Objective
Build a system to predict the predominating type of forest cover using analysis data for 30m x 30m patches of land. This application uses cartographic variables rather than directly sensed imagery.

## Dataset
The dataset from the Roosevelt National Forest of northern Colorado includes cartographic variables such as elevation, slope, distances to hydrology, roadways, fire ignition points, wilderness areas, and soil types.
The target variable is partitioned into 7 integer categories defining the specific Forest Cover Type.

## Features
*   **Machine Learning Model:** Utilizes an optimized Random Forest Classifier capable of discovering non-linear patterns across numerous categorical and continuous mapping variables.
*   **Web Interface:** A Streamlit page allowing users to configure cartographic features manually to instantly view the prediction model's evaluation of the forest setting.

## How to Run Locally

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Files in this Repository
*   `Forest_Cover_Prediction.ipynb`: Model training, EDA, and export.
*   `app.py`: The user-facing Streamlit application.
*   `forest_cover_rf_model.pkl`: The serialized Random Forest model.
*   `Forest_Cover_Prediction_Report.pdf`: Documented technical report.
*   `requirements.txt`: Python package requirements.
