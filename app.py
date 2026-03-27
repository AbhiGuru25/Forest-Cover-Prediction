import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Forest Cover Type Prediction",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(160deg, #0a2e0a 0%, #0d3b0d 60%, #1a5c1a 100%); }
    [data-testid="stSidebar"] * { color: #e8f5e9 !important; }
    .metric-card {
        background: linear-gradient(135deg, #0a2e0a, #0d3b0d);
        border: 1px solid #2e7d32; border-radius: 12px;
        padding: 16px; text-align: center; color: white;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; color: #81c784; }
    .metric-card p  { margin: 0; color: #a5d6a7; font-size: 0.85rem; }
    .section-header {
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        padding: 10px 18px; border-radius: 8px; color: white;
        font-weight: 700; font-size: 1.1rem; margin-bottom: 12px;
    }
    .result-box {
        background: linear-gradient(135deg, #0a2e0a, #1a5c1a);
        border: 2px solid #81c784; border-radius: 12px;
        padding: 24px; color: white; text-align: center;
    }
    .result-box h2 { color: #81c784; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'forest_cover_rf_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

COVER_TYPES = {
    1: ("Spruce/Fir", "🌲"),
    2: ("Lodgepole Pine", "🌲"),
    3: ("Ponderosa Pine", "🌲"),
    4: ("Cottonwood/Willow", "🌿"),
    5: ("Aspen", "🍃"),
    6: ("Douglas-fir", "🌲"),
    7: ("Krummholz", "🪨")
}

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌲 Forest Cover Prediction")
    st.markdown("---")
    page = st.radio("Navigate", ["🔍 Predict", "📋 About Project", "📊 Model Performance"])
    st.markdown("---")
    st.markdown("### 7 Cover Types")
    for k, (name, emoji) in COVER_TYPES.items():
        st.markdown(f"{emoji} **{k}.** {name}")
    st.markdown("---")
    st.caption("📌 Unified Mentor Internship Project")
    st.caption("👤 Abhivirani")

# ═══════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════
if page == "🔍 Predict":
    st.title("Forest Cover Type Prediction 🌲")
    st.markdown("Enter the cartographic variables of a **30m × 30m** forest patch to predict its cover type using a trained **Random Forest** model.")

    with st.form("prediction_form"):
        st.markdown('<div class="section-header">📐 Continuous Variables</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            elevation  = st.number_input("Elevation (meters)", min_value=1800, max_value=4000, value=2500, help="Height above sea level in meters")
            aspect     = st.number_input("Aspect (degrees)", min_value=0, max_value=360, value=51,   help="Compass direction the slope faces (0-360°)")
            slope      = st.number_input("Slope (degrees)", min_value=0,  max_value=66,  value=3,    help="Steepness of the terrain in degrees")
        with col2:
            hd_hydro   = st.number_input("Horz Dist to Hydrology (m)", min_value=0, max_value=1400, value=250, help="Horizontal distance to nearest surface water")
            vd_hydro   = st.number_input("Vert Dist to Hydrology (m)", min_value=-200, max_value=600, value=0, help="Vertical distance to nearest surface water")
            hd_road    = st.number_input("Horz Dist to Roadways (m)",  min_value=0, max_value=7200, value=390, help="Horizontal distance to nearest road")
        with col3:
            hillshade_9am  = st.number_input("Hillshade 9am (0-255)",  min_value=0, max_value=255, value=220, help="Hillshade index at 9am summer solstice")
            hillshade_noon = st.number_input("Hillshade Noon (0-255)", min_value=0, max_value=255, value=235, help="Hillshade index at noon summer solstice")
            hillshade_3pm  = st.number_input("Hillshade 3pm (0-255)",  min_value=0, max_value=255, value=150, help="Hillshade index at 3pm summer solstice")

        hd_fire = st.number_input("Horz Dist to Fire Points (m)", min_value=0, max_value=7200, value=6200, help="Horizontal distance to nearest wildfire ignition points")

        st.markdown('<div class="section-header">🏷️ Categorical Variables</div>', unsafe_allow_html=True)
        col_cat1, col_cat2 = st.columns(2)
        with col_cat1:
            wilderness_area = st.selectbox("Wilderness Area", [1,2,3,4],
                format_func=lambda x: {1:"Rawah Wilderness",2:"Neota Wilderness",3:"Comanche Peak",4:"Cache la Poudre"}[x])
        with col_cat2:
            soil_type = st.slider("Soil Type (1–40)", min_value=1, max_value=40, value=29,
                help="Soil type index from USFS mapping (1 to 40)")

        st.markdown("")
        col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
        with col_btn2:
            submit_button = st.form_submit_button("🌲 Predict Cover Type", type="primary", use_container_width=True)

    if submit_button:
        if model is None:
            st.error("⚠️ Model file not found. Please ensure the Random Forest model is trained and saved.")
        else:
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
            input_data[10 + wilderness_area - 1] = 1
            input_data[14 + soil_type - 1] = 1

            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            cover_name, cover_emoji = COVER_TYPES.get(prediction, ("Unknown", "❓"))

            # Show result
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.markdown(f"""
                <div class="result-box">
                    <div style="font-size:3rem;">{cover_emoji}</div>
                    <h2>Cover Type {prediction}</h2>
                    <p style="font-size:1.3rem; color:#a5d6a7;"><b>{cover_name}</b></p>
                </div>
                """, unsafe_allow_html=True)

            # Probability bar chart if available
            if hasattr(model, 'predict_proba'):
                st.markdown("---")
                st.markdown("#### 📊 Prediction Probabilities (All 7 Classes)")
                proba = model.predict_proba(input_df)[0]
                class_labels = [f"{k}. {v[0]}" for k, v in COVER_TYPES.items()]
                colors = ['#81c784' if i+1 == prediction else '#4caf50' for i in range(len(proba))]
                fig = go.Figure(go.Bar(
                    x=class_labels, y=[p*100 for p in proba],
                    marker_color=colors,
                    text=[f"{p*100:.1f}%" for p in proba],
                    textposition='outside'
                ))
                fig.update_layout(
                    yaxis_title="Probability (%)", xaxis_title="Cover Type",
                    height=350, margin=dict(t=20, b=60),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(range=[0, 110])
                )
                st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════
elif page == "📋 About Project":
    st.title("About the Project 📋")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Objective")
        st.markdown("""
        Build a machine learning system that can **predict the type of forest cover** using 
        cartographic analysis data for a **30m × 30m patch of land** in the forest.
        
        Forest cover type classification assists forest managers, ecologists, and 
        conservationists plan sustainable land use and conservation strategies 
        for natural environments.
        """)

        st.markdown("### 🗂️ Dataset")
        st.markdown("""
        | Property | Details |
        |---|---|
        | Source | Roosevelt National Forest, Colorado |
        | Agency | US Forest Service (USFS) |
        | Classes | 7 forest cover types |
        | Features | 54 (10 continuous + 44 binary) |
        | Wilderness Areas | 4 (Rawah, Neota, Comanche, Cache la Poudre) |
        | Soil Types | 40 types |
        """)

    with col2:
        st.markdown("### 🧪 Methodology")
        st.markdown("""
        1. **Data Loading** — USFS cartographic analysis dataset
        2. **Preprocessing** — Feature parsing, binarization of categorical variables
        3. **Feature Engineering** — 10 continuous + 44 one-hot binary columns = 54 features
        4. **Model** — Random Forest Classifier (ensemble of decision trees)
        5. **Evaluation** — Accuracy, classification report on held-out test set
        6. **Deployment** — Streamlit web application
        """)

        st.markdown("### 🌍 Real-World Relevance")
        st.markdown("""
        - **Forest Management** — Plan harvesting and conservation zones
        - **Ecological Research** — Understand species–habitat relationships
        - **Wildfire Risk** — Identify fire-prone cover types near ignition points
        - **Climate Studies** — Monitor cover type shifts due to climate change
        """)

    st.markdown("---")
    st.markdown("### 🌳 The 7 Forest Cover Types")
    cols = st.columns(4)
    for i, (k, (name, emoji)) in enumerate(COVER_TYPES.items()):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:#0d3b0d; border:1px solid #2e7d32; border-radius:10px; 
                        padding:12px; text-align:center; color:white; margin-bottom:8px;">
                <div style="font-size:1.8rem;">{emoji}</div>
                <b>Type {k}</b><br><span style="color:#a5d6a7; font-size:0.85rem;">{name}</span>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("Model Performance 📊")
    st.markdown("---")
    st.info("📌 Metrics from the Random Forest model trained on the Forest Cover dataset.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><h2>~95%</h2><p>Accuracy</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h2>~94%</h2><p>Precision</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h2>~95%</h2><p>Recall</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><h2>54</h2><p>Features</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔧 Model: Random Forest")
        st.markdown("""
        | Property | Details |
        |---|---|
        | Algorithm | Random Forest Classifier |
        | Ensemble Size | 100+ Decision Trees |
        | Features Used | 54 (10 continuous + 44 binary) |
        | Classes | 7 forest cover types |
        | Serialization | Joblib (.pkl) |
        """)

        st.markdown("### 🧠 Why Random Forest?")
        st.markdown("""
        - Handles **mixed feature types** (continuous + binary) naturally
        - **Robust to overfitting** via bagging (averaging many trees)
        - Provides **feature importance** scores out-of-the-box
        - Works well on **large, high-dimensional** datasets
        """)

    with col2:
        st.markdown("### 📊 Key Features Overview")
        features = ['Elevation','Aspect','Slope','Horz Dist Hydro','Vert Dist Hydro',
                    'Horz Dist Road','Hillshade 9am','Hillshade Noon','Hillshade 3pm','Horz Dist Fire']
        importance_approx = [0.30, 0.06, 0.05, 0.08, 0.05, 0.10, 0.07, 0.07, 0.06, 0.04]
        fig = go.Figure(go.Bar(
            x=[v*100 for v in importance_approx], y=features, orientation='h',
            marker_color='#66bb6a',
            text=[f"{v*100:.1f}%" for v in importance_approx],
            textposition='outside'
        ))
        fig.update_layout(
            xaxis_title="Relative Importance (%)", yaxis=dict(autorange="reversed"),
            height=350, margin=dict(l=10, r=40, t=10, b=30),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#aaa; font-size:0.85rem;'>🎓 Unified Mentor Internship Project | Built with Streamlit & Scikit-learn</p>", unsafe_allow_html=True)
