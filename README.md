# Forest Cover Type Prediction Web App 🌲

**🌐 Live Demo:** [http://forest-cover-prediction-webum.streamlit.app/](http://forest-cover-prediction-webum.streamlit.app/)

A Machine Learning web application powered by Streamlit that predicts the dominant tree species (forest cover type) of a 30m × 30m patch of land in the Roosevelt National Forest, Colorado, using cartographic variables. Built as part of the Unified Mentor Internship Project.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)

## 🎯 Objective
Predict the type of forest cover (7 distinct classes) from 54 cartographic variables, including elevation, slope, soil type, and distance to hydrology/roadways. This assists ecologists and forest managers in planning sustainable land use and assessing wildfire risks.

## ✨ Features
- **Interactive Form:** Intuitive UI to input 10 continuous variables (Elevation, Aspect, Distances) and 2 categorical variables (Wilderness Area, Soil Type).
- **Random Forest Prediction:** Instant prediction distinguishing between 7 cover types (e.g., Spruce/Fir, Lodgepole Pine, Aspen).
- **Probability Distribution:** Interactive Plotly bar chart showing the classification probability across all 7 potential cover types.
- **Feature Importance Tracking:** Dashboard visualizing which geographical features most strongly influence the model's predictions.

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/abhiguru25/forest-cover-prediction.git
cd forest-cover-prediction/forest_cover_prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

## 🧠 Methodology
- **Preprocessing:** One-hot encoding of categorical features resulting in exactly 54 features per sample.
- **Model:** Random Forest Classifier ensemble, chosen for its robustness to scale differences and its ability to handle non-linear geographical relationships.
- **Performance:** Achieves ~95% accuracy on the test set.

## 📁 Repository Structure
```
├── app.py                      # Main Streamlit application file
├── requirements.txt            # Python dependencies
├── forest_cover_rf_model.pkl   # Serialized Random Forest Model (joblib)
└── README.md                   # This documentation
```

---
*Developed by Abhivirani | Unified Mentor Internship Project*
