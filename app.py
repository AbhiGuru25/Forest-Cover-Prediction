import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Forest Cover Type Prediction", page_icon="🌲")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'forest_cover_rf_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# Forest cover types 1 to 7
cover_type_mapping = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

st.title("Forest Cover Type Prediction 🌲")
st.write("Enter the cartographic variables of a 30m x 30m forest patch to predict its cover type.")

with st.form("prediction_form"):
    st.subheader("Continuous Variables")
    col1, col2, col3 = st.columns(3)
    with col1:
        elevation = st.number_input("Elevation (meters)", min_value=1800, max_value=4000, value=2500)
        aspect = st.number_input("Aspect (degrees)", min_value=0, max_value=360, value=51)
        slope = st.number_input("Slope (degrees)", min_value=0, max_value=66, value=3)
    with col2:
        hd_hydro = st.number_input("Horz Dist to Hydrology", min_value=0, max_value=1400, value=250)
        vd_hydro = st.number_input("Vert Dist to Hydrology", min_value=-200, max_value=600, value=0)
        hd_road = st.number_input("Horz Dist to Roadways", min_value=0, max_value=7200, value=390)
    with col3:
        hillshade_9am = st.number_input("Hillshade 9am (0-255)", min_value=0, max_value=255, value=220)
        hillshade_noon = st.number_input("Hillshade Noon (0-255)", min_value=0, max_value=255, value=235)
        hillshade_3pm = st.number_input("Hillshade 3pm (0-255)", min_value=0, max_value=255, value=150)
    
    hd_fire = st.number_input("Horz Dist to Fire Points", min_value=0, max_value=7200, value=6200)

    st.subheader("Categorical Variables")
    wilderness_area = st.selectbox("Wilderness Area", [1, 2, 3, 4], format_func=lambda x: f"Area {x}")
    soil_type = st.slider("Soil Type (1-40)", min_value=1, max_value=40, value=29)

    submit_button = st.form_submit_button("Predict Cover Type")

if submit_button:
    if model is None:
        st.error("Model file (forest_cover_rf_model.pkl) not found. Please ensure it is trained and saved in this directory.")
    else:
        # Create input array
        input_data = np.zeros(54)
        input_data[0] = elevation
        input_data[1] = aspect
        input_data[2] = slope
        input_data[3] = hd_hydro
        input_data[4] = vd_hydro
        input_data[5] = hd_road
        input_data[6] = hillshade_9am
        input_data[7] = hillshade_noon
        input_data[8] = hillshade_3pm
        input_data[9] = hd_fire
        
        # Wilderness Area (4 binary columns, idx 10 to 13)
        input_data[10 + wilderness_area - 1] = 1
        
        # Soil Type (40 binary columns, idx 14 to 53)
        input_data[14 + soil_type - 1] = 1
        
        # Reshape for prediction
        input_df = pd.DataFrame([input_data])
        
        prediction = model.predict(input_df)[0]
        cover_name = cover_type_mapping.get(prediction, "Unknown")
        
        st.success(f"**Predicted Cover Type:** {prediction} - {cover_name}")
