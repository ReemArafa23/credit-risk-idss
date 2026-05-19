import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(page_title="IDSS Active Decision Engine", layout="wide")

# 1. Architecture & Setup
@st.cache_resource
def load_model_and_features():
    model = joblib.load('best_credit_risk_model.joblib')
    features = joblib.load('model_features.joblib')
    return model, features

model, features = load_model_and_features()

st.title("🏦 Intelligent Decision Support System (IDSS)")
st.markdown("### Phase 5: Active Decision Engine Prototype")

# Sidebar - Applicant Details
st.sidebar.header("Applicant Details")
age = st.sidebar.slider("Age", 18, 100, 30)
income = st.sidebar.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
emp_length = st.sidebar.slider("Employment Length (Years)", 0, 50, 5)
loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=0, value=15000, step=500)
int_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 25.0, 10.5)
cred_hist_length = st.sidebar.slider("Credit History Length (Years)", 0, 30, 4)

home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
prior_default = st.sidebar.selectbox("Prior Default on File", ["No", "Yes"])

if st.sidebar.button("Assess Risk", type="primary"):
    
    # 2. Dynamic Feature Engineering
    loan_to_income_ratio = loan_amnt / income if income > 0 else 0
    emp_length_to_age_ratio = emp_length / age if age > 0 else 0
    
    # Initialize all expected features to 0.0 to guarantee schema match
    input_data = {feat: 0.0 for feat in features}
    
    # Safely map inputs to the exact model columns
    if 'person_age' in input_data: input_data['person_age'] = age
    if 'person_income' in input_data: input_data['person_income'] = income
    if 'person_emp_length' in input_data: input_data['person_emp_length'] = emp_length
    if 'loan_amnt' in input_data: input_data['loan_amnt'] = loan_amnt
    if 'loan_int_rate' in input_data: input_data['loan_int_rate'] = int_rate
    if 'cb_person_cred_hist_length' in input_data: input_data['cb_person_cred_hist_length'] = cred_hist_length
    
    # Mapping the engineered features
    if 'loan_percent_income' in input_data: input_data['loan_percent_income'] = loan_to_income_ratio
    elif 'loan_to_income_ratio' in input_data: input_data['loan_to_income_ratio'] = loan_to_income_ratio
    if 'emp_length_to_age_ratio' in input_data: input_data['emp_length_to_age_ratio'] = emp_length_to_age_ratio
    
    # Mapping One-Hot Encodings
    home_col = f"person_home_ownership_{home_ownership}"
    if home_col in input_data: input_data[home_col] = 1.0
    
    default_col = "cb_person_default_on_file_Y" if prior_default == "Yes" else "cb_person_default_on_file_N"
    if default_col in input_data: input_data[default_col] = 1.0
    
    applicant_df = pd.DataFrame([input_data], columns=features)
    
    # 3. The IDSS Active Decision Engine
    prob_default = model.predict_proba(applicant_df)[0][1]
    
    if prob_default > 0.7:
        tier_text = "High Risk"
        tier_color = "#FF4B4B" # Red
    elif prob_default >= 0.3:
        tier_text = "Medium Risk"
        tier_color = "#FFA500" # Yellow/Orange
    else:
        tier_text = "Low Risk"
        tier_color = "#00C04B" # Green
        
    st.markdown("---")
    st.subheader("1. IDSS Risk Assessment")
    
    col1, col2 = st.columns(2)
    col1.metric(label="Probability of Default", value=f"{prob_default:.2%}")
    col2.markdown(f"### Risk Tier: <span style='color:{tier_color}'>{tier_text}</span>", unsafe_allow_html=True)
    
    # XAI Extraction
    local_explainer = shap.TreeExplainer(model)
    local_shap_vals = local_explainer(applicant_df)
    
    feature_contributions = dict(zip(features, local_shap_vals.values[0]))
    sorted_drivers = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
    top_2_drivers = [item for item in sorted_drivers if item[1] > 0][:2]
    
    st.markdown("---")
    st.subheader("2. Automated Business Recommendation")
    
    if len(top_2_drivers) == 0:
        st.success("PROCEED: No significant risk drivers identified. Approve under standard protocol.")
    else:
        top_driver = top_2_drivers[0][0].lower()
        
        if 'loan_to_income' in top_driver or 'percent_income' in top_driver:
            st.error("RECOMMENDATION: Counter-offer with a lower principal loan amount to bring the LTI ratio under 35%.")
        elif 'int_rate' in top_driver:
            st.warning("RECOMMENDATION: Reject current unsecured terms, but offer a secured loan option to lower the interest burden.")
        elif 'person_income' in top_driver:
            st.error("RECOMMENDATION: Income is flagged as insufficient for this loan structure. Request proof of additional income or a co-signer.")
        elif 'home_ownership' in top_driver:
            st.warning("RECOMMENDATION: Housing instability flagged. Require a larger down payment or collateral to mitigate flight risk.")
        elif 'cred_hist_length' in top_driver:
            st.warning("RECOMMENDATION: Short credit history driving risk. Require secondary credit references (e.g., utility bills).")
        elif 'emp_length' in top_driver:
            st.warning("RECOMMENDATION: Short or unstable employment history flagged. Require most recent pay stubs and employer verification.")
        elif 'grade' in top_driver:
             st.warning("RECOMMENDATION: Poor historical loan grade detected. Limit total exposure and require auto-pay enrollment.")
        else:
             st.error(f"RECOMMENDATION: Elevated risk stemming from `{top_2_drivers[0][0]}`. Escalate to human underwriter for manual review.")

    # 4. Visual Explainability (Local SHAP)
    st.markdown("---")
    st.subheader("3. Audit Trail: SHAP Local Explainability")
    st.markdown("The waterfall plot below provides mathematical proof of the ML features driving this exact decision.")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(local_shap_vals[0], show=False)
    st.pyplot(fig)