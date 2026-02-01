import streamlit as st
import joblib
import pandas as pd
import os

# Page Configuration
st.set_page_config(
    page_title="Purchase Prediction System",
    page_icon="🛒",
    layout="centered"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    .stSuccess, .stWarning {
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("🛒 Purchase Prediction System")
st.markdown("Predict whether a customer will make a purchase based on their behavior and demographics.")
st.markdown("---")

# Sidebar for Model Selection
st.sidebar.header("Model Configuration")

models_dir = 'models'
model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and 'kmeans' not in f] if os.path.exists(models_dir) else []

if not model_files:
    st.error("No model files found in 'models' directory!")
    selected_model_name = None
else:
    model_display_names = {
        'logistic_no_cluster.pkl': 'Logistic Regression (No Cluster)',
        'logistic_with_cluster.pkl': 'Logistic Regression (With Cluster)',
        # 'random_forest_no_cluster.pkl': 'Random Forest (No Cluster)',
        # 'random_forest_with_cluster.pkl': 'Random Forest (With Cluster)'
    }
    
    # Create a reverse mapping for selection
    # Create a reverse mapping for selection (only include models explicitly listed above)
    display_to_filename = {v: k for k, v in model_display_names.items() if k in model_files}
    
    # Argument: We only want to show models that are explicitly defined in the dictionary.
    # Any other .pkl files in the directory should be ignored to prevent unwanted models (like Random Forest) from appearing.
    # for f in model_files:
    #     if f not in model_display_names:
    #         display_to_filename[f] = f

    selected_display_name = st.sidebar.selectbox(
        "Select Prediction Model",
        options=list(display_to_filename.keys())
    )
    selected_model_name = display_to_filename[selected_display_name]

# Function to load model
@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(models_dir, model_name)
    
    if not os.path.exists(model_path):
        return None
        
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

if selected_model_name:
    model = load_model(selected_model_name)
    
    if model:
        # Determine if cluster feature is needed
        exclude_cluster = "no_cluster" in selected_model_name

        # User Input Interface
        st.subheader("Customer Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 100, 30)
            annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
            purchases = st.slider("Number of Purchases", 0, 100, 5)
            time_spent = st.slider("Time Spent on Website (mins)", 0.0, 300.0, 30.0)
            tenure = st.slider("Customer Tenure (Years)", 0.0, 20.0, 2.0)
            
        with col2:
            last_purchase = st.slider("Days Since Last Purchase", 0, 365, 30)
            discounts = st.slider("Discounts Availed", 0, 50, 2)
            sessions = st.slider("Session Count", 1, 50, 5)
            satisfaction = st.slider("Customer Satisfaction (1-5)", 1, 5, 3)
            loyalty = st.selectbox("Loyalty Program Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
            cluster_value = 0
            if not exclude_cluster:
                cluster_value = st.selectbox("Customer Cluster", [0, 1, 2, 3])
            else:
                st.info("Cluster feature not used for this model.")

        # Prediction Logic
        if st.button("Predict Purchase"):
            # Create feature dictionary with exact names matching training data
            input_data = {
                'Age': [age],
                'AnnualIncome': [annual_income],
                'NumberOfPurchases': [purchases],
                'TimeSpentOnWebsite': [time_spent],
                'CustomerTenureYears': [tenure],
                'LastPurchaseDaysAgo': [last_purchase],
                'DiscountsAvailed': [discounts],
                'SessionCount': [sessions],
                'CustomerSatisfaction': [satisfaction],
                'LoyaltyProgram': [loyalty]
            }
            
            if not exclude_cluster:
                input_data['Cluster'] = [cluster_value]
                
            # Create DataFrame
            input_df = pd.DataFrame(input_data)
            
            try:
                # Make prediction using the pipeline (includes scaling)
                prediction = model.predict(input_df)[0]
                
                probability = 0.0
                if hasattr(model, "predict_proba"):
                    probability = model.predict_proba(input_df)[0][1]
                
                st.markdown("---")
                st.subheader("Prediction Result")
                
                col_res1, col_res2 = st.columns([2, 1])
                
                with col_res1:
                    if prediction == 1:
                        st.success(f"### Likely to Purchase")
                        if hasattr(model, "predict_proba"):
                            st.markdown(f"**Probability:** {probability:.2f}")
                    else:
                        st.warning(f"### Unlikely to Purchase")
                        if hasattr(model, "predict_proba"):
                            st.markdown(f"**Probability:** {probability:.2f}")
                        
                with col_res2:
                    if hasattr(model, "predict_proba"):
                        st.metric("Confidence Score", f"{probability:.2%}")
                        st.progress(probability)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Ensure the feature names in the app match exactly what the model was trained on.")

    else:
        st.error(f"Failed to load model: {selected_model_name}")
