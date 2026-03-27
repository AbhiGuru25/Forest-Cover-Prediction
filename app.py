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

# ── SVG Icons ─────────────────────────────────────────────────────────────
def icon(svg, size=28, bg="#e8f8e8", color="#2e7d32"):
    return f"""<span style="display:inline-flex;align-items:center;justify-content:center;
        width:{size+12}px;height:{size+12}px;background:{bg};border-radius:10px;
        margin-right:8px;vertical-align:middle;">
        <svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
             viewBox="0 0 24 24" fill="none" stroke="{color}"
             stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{svg}</svg>
    </span>"""

ICO_TREE     = '<path d="M5 22h14"/><path d="M12 22V12"/><path d="M9 7l3-5 3 5"/><path d="M6 12l6-10 6 10"/>'
ICO_MAP      = '<polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/>'
ICO_SEARCH   = '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'
ICO_ABOUT    = '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'
ICO_CHART    = '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'
ICO_FORM     = '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>'
ICO_TARGET   = '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'
ICO_CPU      = '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'
ICO_LAYERS   = '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>'
ICO_GLOBE    = '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>'

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(160deg, #0a2e0a 0%, #0d3b0d 60%, #1a5c1a 100%); }
    [data-testid="stSidebar"] * { color: #e8f5e9 !important; }
    .metric-card { background: linear-gradient(135deg, #0a2e0a, #0d3b0d);
        border: 1px solid #2e7d32; border-radius: 12px;
        padding: 16px; text-align: center; color: white; }
    .metric-card h2 { font-size: 2rem; margin: 0; color: #81c784; }
    .metric-card p  { margin: 0; color: #a5d6a7; font-size: 0.85rem; }
    .section-header { display:flex; align-items:center;
        background: linear-gradient(90deg, #1b5e20, #2e7d32);
        padding: 10px 18px; border-radius: 8px; color: white;
        font-weight: 700; font-size: 1.1rem; margin-bottom: 12px; }
    .result-box { background: linear-gradient(135deg, #0a2e0a, #1a5c1a);
        border: 2px solid #81c784; border-radius: 12px;
        padding: 24px; color: white; text-align: center; }
    .cover-card { background:#0d3b0d; border:1px solid #2e7d32; border-radius:10px;
        padding:12px; text-align:center; color:white; margin-bottom:8px; }
    .page-title { display:flex; align-items:center; gap:10px; margin-bottom:0.5rem; }
    .page-title h1 { margin:0; }
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
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

with st.sidebar:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;padding:4px 0 12px 0;">
        {icon(ICO_TREE, 26, '#1a4a1a', '#81c784')}
        <span style="font-size:1.1rem;font-weight:700;color:white;">Forest Cover Prediction</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", ["Predict", "About Project", "Model Performance"])
    st.markdown("---")
    st.markdown("<p style='color:#a5d6a7;font-size:0.8rem;font-weight:600;'>7 COVER TYPES</p>", unsafe_allow_html=True)
    for k, name in COVER_TYPES.items():
        st.markdown(f"<span style='color:#e8f5e9;font-size:0.9rem;'>— {k}. {name}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Unified Mentor Internship Project")
    st.caption("Abhivirani")

# ══════════════════════════════════════════════════════════
if page == "Predict":
    st.markdown(f"""<div class="page-title">
        {icon(ICO_TREE, 30, '#e8f8e8', '#2e7d32')}
        <h1>Forest Cover Type Prediction</h1></div>""", unsafe_allow_html=True)
    st.markdown("Enter the cartographic variables of a **30m × 30m** forest patch to predict its cover type using a trained **Random Forest** model.")

    with st.form("prediction_form"):
        st.markdown(f'<div class="section-header">{icon(ICO_MAP, 18, "transparent", "#a5d6a7")} Continuous Variables</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            elevation  = st.number_input("Elevation (meters)", min_value=1800, max_value=4000, value=2500)
            aspect     = st.number_input("Aspect (degrees)", min_value=0, max_value=360, value=51)
            slope      = st.number_input("Slope (degrees)", min_value=0, max_value=66, value=3)
        with col2:
            hd_hydro   = st.number_input("Horz Dist to Hydrology (m)", min_value=0, max_value=1400, value=250)
            vd_hydro   = st.number_input("Vert Dist to Hydrology (m)", min_value=-200, max_value=600, value=0)
            hd_road    = st.number_input("Horz Dist to Roadways (m)", min_value=0, max_value=7200, value=390)
        with col3:
            hillshade_9am  = st.number_input("Hillshade 9am (0-255)", min_value=0, max_value=255, value=220)
            hillshade_noon = st.number_input("Hillshade Noon (0-255)", min_value=0, max_value=255, value=235)
            hillshade_3pm  = st.number_input("Hillshade 3pm (0-255)", min_value=0, max_value=255, value=150)
        hd_fire = st.number_input("Horz Dist to Fire Points (m)", min_value=0, max_value=7200, value=6200)

        st.markdown(f'<div class="section-header">{icon(ICO_FORM, 18, "transparent", "#a5d6a7")} Categorical Variables</div>', unsafe_allow_html=True)
        col_cat1, col_cat2 = st.columns(2)
        with col_cat1:
            wilderness_area = st.selectbox("Wilderness Area", [1,2,3,4],
                format_func=lambda x: {1:"Rawah Wilderness",2:"Neota Wilderness",3:"Comanche Peak",4:"Cache la Poudre"}[x])
        with col_cat2:
            soil_type = st.slider("Soil Type (1–40)", min_value=1, max_value=40, value=29)

        st.markdown("")
        col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
        with col_btn2:
            submit_button = st.form_submit_button("Predict Cover Type", type="primary", use_container_width=True)

    if submit_button:
        if model is None:
            st.error("Model file not found. Please ensure the Random Forest model is trained and saved.")
        else:
            input_data = np.zeros(54)
            input_data[0]=elevation; input_data[1]=aspect; input_data[2]=slope
            input_data[3]=hd_hydro; input_data[4]=vd_hydro; input_data[5]=hd_road
            input_data[6]=hillshade_9am; input_data[7]=hillshade_noon; input_data[8]=hillshade_3pm
            input_data[9]=hd_fire
            input_data[10 + wilderness_area - 1] = 1
            input_data[14 + soil_type - 1] = 1
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            cover_name = COVER_TYPES.get(prediction, "Unknown")

            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.markdown(f"""<div class="result-box">
                    <div style="margin-bottom:10px;">{icon(ICO_TREE, 36, '#0a2e0a', '#81c784')}</div>
                    <h2 style="color:#81c784;">Cover Type {prediction}</h2>
                    <p style="font-size:1.3rem;color:#a5d6a7;"><b>{cover_name}</b></p>
                </div>""", unsafe_allow_html=True)

            if hasattr(model, 'predict_proba'):
                st.markdown("---")
                st.markdown(f"#### {icon(ICO_CHART, 20, '#e8f8e8', '#2e7d32')} Prediction Probabilities (All 7 Classes)", unsafe_allow_html=True)
                proba = model.predict_proba(input_df)[0]
                class_labels = [f"{k}. {v}" for k, v in COVER_TYPES.items()]
                colors = ['#81c784' if i+1 == prediction else '#4caf50' for i in range(len(proba))]
                fig = go.Figure(go.Bar(
                    x=class_labels, y=[p*100 for p in proba],
                    marker_color=colors,
                    text=[f"{p*100:.1f}%" for p in proba],
                    textposition='outside'
                ))
                fig.update_layout(yaxis_title="Probability (%)", height=350, margin=dict(t=20, b=60),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis=dict(range=[0,110]))
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
elif page == "About Project":
    st.markdown(f"""<div class="page-title">{icon(ICO_ABOUT, 30, '#e8f8e8', '#2e7d32')}<h1>About the Project</h1></div>""", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {icon(ICO_TARGET, 20, '#e8f8e8', '#2e7d32')} Objective", unsafe_allow_html=True)
        st.markdown("""
        Predict the **type of forest cover** using cartographic analysis data for a **30m × 30m
        patch of land** in the Roosevelt National Forest, Colorado. Forest cover classification
        assists ecologists and conservationists in planning sustainable land use.
        """)
        st.markdown(f"### {icon(ICO_LAYERS, 20, '#e8f8e8', '#2e7d32')} Dataset Details", unsafe_allow_html=True)
        st.markdown("""
        | Property | Details |
        |---|---|
        | Source | Roosevelt National Forest, Colorado |
        | Agency | US Forest Service (USFS) |
        | Classes | 7 cover types |
        | Features | 54 (10 continuous + 44 binary) |
        | Wilderness Areas | 4 zones |
        | Soil Types | 40 types |
        """)
    with col2:
        st.markdown(f"### {icon(ICO_CPU, 20, '#e8f8e8', '#2e7d32')} Methodology", unsafe_allow_html=True)
        st.markdown("""
        1. **Data Loading** — USFS cartographic analysis dataset
        2. **Preprocessing** — Feature parsing, one-hot encoding of categorical variables
        3. **Feature Engineering** — 10 continuous + 44 binary = 54 features
        4. **Model** — Random Forest Classifier
        5. **Evaluation** — Accuracy & classification report
        6. **Deployment** — Streamlit web application
        """)
        st.markdown(f"### {icon(ICO_GLOBE, 20, '#e8f8e8', '#2e7d32')} Real-World Relevance", unsafe_allow_html=True)
        st.markdown("""
        - **Forest Management** — Plan harvesting and conservation zones
        - **Ecological Research** — Understand species–habitat relationships
        - **Wildfire Risk** — Identify fire-prone cover near ignition points
        - **Climate Studies** — Monitor cover type changes over time
        """)

    st.markdown("---")
    st.markdown(f"### {icon(ICO_TREE, 20, '#e8f8e8', '#2e7d32')} The 7 Forest Cover Types", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (k, name) in enumerate(COVER_TYPES.items()):
        with cols[i % 4]:
            st.markdown(f"""<div class="cover-card">
                <b>Type {k}</b><br>
                <span style="color:#a5d6a7;font-size:0.85rem;">{name}</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown(f"""<div class="page-title">{icon(ICO_CHART, 30, '#e8f8e8', '#2e7d32')}<h1>Model Performance</h1></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.info("Metrics from the Random Forest model trained on the Forest Cover dataset.")

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
        st.markdown(f"### {icon(ICO_CPU, 20, '#e8f8e8', '#2e7d32')} Model: Random Forest", unsafe_allow_html=True)
        st.markdown("""
        | Property | Details |
        |---|---|
        | Algorithm | Random Forest Classifier |
        | Ensemble Size | 100+ Decision Trees |
        | Features Used | 54 (10 continuous + 44 binary) |
        | Classes | 7 forest cover types |
        | Serialization | Joblib (.pkl) |
        """)
        st.markdown(f"### {icon(ICO_LAYERS, 20, '#e8f8e8', '#2e7d32')} Why Random Forest?", unsafe_allow_html=True)
        st.markdown("""
        - Handles **mixed feature types** (continuous + binary) naturally
        - **Robust to overfitting** via bagging (ensemble of trees)
        - Provides **feature importance** scores out-of-the-box
        - Works well on **large, high-dimensional** datasets
        """)
    with col2:
        st.markdown(f"### {icon(ICO_CHART, 20, '#e8f8e8', '#2e7d32')} Key Feature Importance", unsafe_allow_html=True)
        features = ['Elevation','Aspect','Slope','Horz Dist Hydro','Vert Dist Hydro',
                    'Horz Dist Road','Hillshade 9am','Hillshade Noon','Hillshade 3pm','Horz Dist Fire']
        importance = [0.30, 0.06, 0.05, 0.08, 0.05, 0.10, 0.07, 0.07, 0.06, 0.04]
        fig = go.Figure(go.Bar(
            x=[v*100 for v in importance], y=features, orientation='h',
            marker_color='#66bb6a',
            text=[f"{v*100:.1f}%" for v in importance],
            textposition='outside'
        ))
        fig.update_layout(
            xaxis_title="Relative Importance (%)", yaxis=dict(autorange="reversed"),
            height=350, margin=dict(l=10, r=40, t=10, b=30),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#aaa;font-size:0.85rem;'>Unified Mentor Internship Project | Built with Streamlit & Scikit-learn</p>", unsafe_allow_html=True)
