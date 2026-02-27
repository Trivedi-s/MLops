import streamlit as st
import requests

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS FOR STYLING
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF2E2E;
        border-color: #FF2E2E;
    }
    div[data-testid="stMetric"] {
    background-color: #262730;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}
    </style>
""", unsafe_allow_html=True)

# 3. HELPER FUNCTION TO NORMALIZE VALUES
# These normalization values are approximate based on the sklearn diabetes dataset
def normalize(value, mean, std):
    return (value - mean) / std

# Approximate means and stds from the original diabetes dataset
NORMALIZATION = {
    'age': {'mean': 48.5, 'std': 13.1},      # years
    'bmi': {'mean': 26.4, 'std': 4.4},       # kg/m²
    'bp': {'mean': 94.6, 'std': 13.8},       # mm Hg
    's1': {'mean': 189.1, 'std': 34.6},      # tc (total cholesterol)
    's2': {'mean': 115.4, 'std': 30.4},      # ldl
    's3': {'mean': 49.8, 'std': 12.9},       # hdl
    's4': {'mean': 4.1, 'std': 1.3},         # tch (thyroid)
    's5': {'mean': 4.6, 'std': 0.5},         # ltg (log of triglycerides)
    's6': {'mean': 91.3, 'std': 11.5},       # glu (glucose)
}

# 4. SIDEBAR CONFIGURATION
with st.sidebar:
    st.title("🩺 About the App")
    st.info(
        """
        This machine learning app predicts **diabetes disease progression** based on patient measurements.
        
        **Input Variables:**
        - Age, Sex, BMI, Blood Pressure
        - Blood tests: Cholesterol, LDL, HDL, etc.
        
        **Output:** A score (25-350) indicating disease progression one year after baseline.
        - **Low (<100):** Minimal progression
        - **Medium (100-200):** Moderate progression  
        - **High (>200):** Significant progression
        """
    )
    st.write("---")
    st.caption("Built with Streamlit & Cloud Run")

# 5. MAIN APP INTERFACE
st.title("🩺 Diabetes Progression Predictor")
st.markdown("Enter patient information below to predict disease progression.")

# Create a grid for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Info")
    age = st.number_input('Age (years)', min_value=18, max_value=90, value=50)
    sex_choice = st.selectbox('Sex', ['Female', 'Male'])
    bmi = st.number_input('BMI (kg/m²)', min_value=15.0, max_value=50.0, value=26.0, step=0.1)
    bp = st.number_input('Blood Pressure (mm Hg)', min_value=60, max_value=160, value=95)
    
with col2:
    st.subheader("Blood Tests")
    s1 = st.number_input('Total Cholesterol (mg/dL)', min_value=100, max_value=300, value=190)
    s2 = st.number_input('LDL Cholesterol (mg/dL)', min_value=50, max_value=250, value=115)
    s3 = st.number_input('HDL Cholesterol (mg/dL)', min_value=20, max_value=100, value=50)
    s4 = st.number_input('Cholesterol/HDL Ratio', min_value=1.0, max_value=10.0, value=4.0, step=0.1)
    s5 = st.number_input('Log Triglycerides', min_value=3.0, max_value=6.5, value=4.6, step=0.1)
    s6 = st.number_input('Blood Sugar (mg/dL)', min_value=60, max_value=150, value=91)

st.write("---")

# 6. PREDICTION LOGIC
if st.button('🔍 Predict Progression'):
    
    with st.spinner('Analyzing patient data...'):
        
        # Normalize all values
        age_norm = normalize(age, **NORMALIZATION['age'])
        sex_norm = -0.044 if sex_choice == 'Female' else 0.050
        bmi_norm = normalize(bmi, **NORMALIZATION['bmi'])
        bp_norm = normalize(bp, **NORMALIZATION['bp'])
        s1_norm = normalize(s1, **NORMALIZATION['s1'])
        s2_norm = normalize(s2, **NORMALIZATION['s2'])
        s3_norm = normalize(s3, **NORMALIZATION['s3'])
        s4_norm = normalize(s4, **NORMALIZATION['s4'])
        s5_norm = normalize(s5, **NORMALIZATION['s5'])
        s6_norm = normalize(s6, **NORMALIZATION['s6'])
        
        data = {
            'age': age_norm,
            'sex': sex_norm,
            'bmi': bmi_norm,
            'bp': bp_norm,
            's1': s1_norm,
            's2': s2_norm,
            's3': s3_norm,
            's4': s4_norm,
            's5': s5_norm,
            's6': s6_norm
        }
        
        try:
            response = requests.post('https://diabetes-app-867291392291.us-east1.run.app/predict', json=data)
            
            if response.status_code == 200:
                prediction = response.json()['prediction']
                
                st.success("Prediction Complete!")
                
                # Display result with large metric
                col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
                with col_result2:
                    st.metric(label="Disease Progression Score", value=f"{prediction}")
                
                # Interpretation with progress bar
                st.write("")
                progress_value = min(prediction / 350, 1.0)  # Normalize to 0-1
                st.progress(progress_value)
                
                # Interpretation message
                if prediction < 100:
                    st.success("📉 **Low Risk** - Below average disease progression expected")
                elif prediction < 200:
                    st.warning("📊 **Moderate Risk** - Average disease progression expected")
                else:
                    st.error("📈 **High Risk** - Above average disease progression expected")
                
                # Show input summary
                with st.expander("View Input Summary"):
                    st.write(f"""
                    **Patient Profile:**
                    - Age: {age} years
                    - Sex: {sex_choice}
                    - BMI: {bmi} kg/m²
                    - Blood Pressure: {bp} mm Hg
                    
                    **Blood Tests:**
                    - Total Cholesterol: {s1} mg/dL
                    - LDL: {s2} mg/dL
                    - HDL: {s3} mg/dL
                    - Cholesterol/HDL Ratio: {s4}
                    - Log Triglycerides: {s5}
                    - Blood Sugar: {s6} mg/dL
                    """)
                
            
            else:
                st.error(f'Server Error: {response.status_code}')
                
        except requests.exceptions.RequestException as e:
            st.error('Connection Error: Could not reach the prediction service.')