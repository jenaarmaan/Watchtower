import streamlit as st

# Import your existing modules
import data_generator
import watchtower
import dashboard

st.set_page_config(
    page_title="Watchtower – Model Risk Early Warning",
    layout="wide"
)

st.title("🛡️ Watchtower")
st.caption("Label-free early warning system for silent model decay")

st.markdown("""
**Philosophy**
- Preventive, not reactive  
- Signal-driven, not accuracy-driven  
- Detects *risk*, not ground-truth correctness  
""")

# Sidebar controls
st.sidebar.header("Simulation Controls")
num_samples = st.sidebar.slider("Number of samples", 100, 5000, 1000)
drift_level = st.sidebar.slider("Drift intensity", 0.0, 1.0, 0.3)

# Generate demo data
data = data_generator.generate_data(
    n=num_samples,
    drift_level=drift_level
)

# Run Watchtower pipeline
results = watchtower.run(data)

# Render dashboard
dashboard.render(results)
