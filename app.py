import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
import plotly.express as px
import plotly.graph_objects as go

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

def show_portfolio_insights():
    st.header("📈 Portfolio Insights & Strategic Dashboard")
    st.markdown("Analyze systemic risk, optimize credit policies, and uncover revenue opportunities across the applicant portfolio.")

    # Data Loading (Checking multiple paths for robustness)
    try:
        df = pd.read_csv('Dataset/X_test_phase3.csv') 
    except FileNotFoundError:
        try:
            df = pd.read_csv('X_test_phase3.csv')
        except FileNotFoundError:
            st.warning("⚠️ Could not locate X_test_phase3.csv. Please ensure the dataset is in the working directory.")
            return

    # Ensure required features are present, handle NaNs
    try:
        df_model = df[features].fillna(0)
    except KeyError as e:
        st.error(f"Missing expected features in dataset: {e}")
        return

    # Calculate probabilities dynamically
    with st.spinner("Calculating portfolio risk..."):
        df['prob_default'] = model.predict_proba(df_model)[:, 1]
    
    # Define Risk Tiers
    bins = [0, 0.3, 0.7, 1.0]
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    df['Risk_Tier'] = pd.cut(df['prob_default'], bins=bins, labels=labels, include_lowest=True)
    
    # Top-Level KPIs
    tot_assessed = len(df)
    tot_value = df['loan_amnt'].sum()
    avg_default = df['prob_default'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Assessments in Portfolio", f"{tot_assessed:,}")
    col2.metric("Total Portfolio Value", f"${tot_value:,.2f}")
    col3.metric("Average Default Probability", f"{avg_default:.2%}")
    st.divider()

    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Risk vs. Return", 
        "🚦 Operational Funnel & Leakage", 
        "📈 Financial Optimization",
        "🗺️ Demographic Exposure"
    ])

    # --- Insight 1: Risk vs. Return (Strategic Profitability) ---
    with tab1:
        st.subheader("1. The Risk vs. Return Scatter Matrix")
        st.markdown("**Executive Summary:** Identifies the 'Gold Mine' (low risk, high margin) versus the 'Danger Zone' (high risk, low margin) to immediately reposition origination strategies.")
        
        # Sample data if too large to prevent lagging
        df_plot = df.sample(n=min(2000, len(df)), random_state=42)
        
        fig1 = px.scatter(
            df_plot, 
            x='prob_default', 
            y='loan_int_rate', 
            size='loan_amnt', 
            color='Risk_Tier',
            color_discrete_map={'Low Risk': '#00C04B', 'Medium Risk': '#FFA500', 'High Risk': '#FF4B4B'},
            hover_name='Risk_Tier',
            title="Strategic Profitability: Return (Interest Rate) vs. Risk (Prob of Default)",
            labels={'prob_default': 'Probability of Default (Risk)', 'loan_int_rate': 'Interest Rate (Return %)'},
            opacity=0.6
        )
        fig1.add_vline(x=0.3, line_dash="dash", line_color="green", annotation_text="Low Risk Boundary")
        fig1.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="High Risk Boundary")
        st.plotly_chart(fig1, use_container_width=True)

    # --- Insight 2 & 4: Operational Funnel & Opportunity Leakage ---
    with tab2:
        st.subheader("2. The Dynamic Approval Funnel (Operational Impact)")
        st.markdown("**Executive Summary:** Simulates policy tightening/loosening in real-time. Adjust the threshold below to see immediate volume impacts.")
        
        threshold = st.slider("Maximum Acceptable Default Probability", min_value=0.05, max_value=0.90, value=0.30, step=0.05)
        
        df['Approval_Status'] = np.where(df['prob_default'] <= threshold, 'Approved', 'Rejected')
        
        funnel_data = df.groupby('Approval_Status')['loan_amnt'].sum().reset_index()
        
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            fig2 = px.pie(
                funnel_data, 
                values='loan_amnt', 
                names='Approval_Status', 
                hole=0.4,
                color='Approval_Status',
                color_discrete_map={'Approved': '#00C04B', 'Rejected': '#FF4B4B'},
                title=f"Loan Volume ($) at {threshold:.0%} Cutoff"
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        with col_f2:
            st.subheader("3. 'Opportunity Leakage' Analysis (Revenue Recovery)")
            st.markdown("**Executive Summary:** Uncovers *why* we are rejecting deals, enabling the creation of targeted down-sell products (e.g., lower limits).")
            
            rejected_df = df[df['Approval_Status'] == 'Rejected']
            approved_df = df[df['Approval_Status'] == 'Approved']
            
            if not rejected_df.empty:
                # Approximate top risk drivers by comparing standardized deviations of top global features
                top_features = ['loan_percent_income', 'loan_int_rate', 'person_income', 'person_emp_length', 'cb_person_cred_hist_length']
                valid_features = [f for f in top_features if f in df.columns]
                
                leakage_data = []
                for f in valid_features:
                    rej_mean = rejected_df[f].mean()
                    app_mean = approved_df[f].mean()
                    # Calculate a simple % difference metric 
                    diff_pct = ((rej_mean - app_mean) / (app_mean + 1e-9)) * 100
                    leakage_data.append({'Feature': f, 'Rejection Driver Severity (%)': abs(diff_pct)})
                
                leakage_df = pd.DataFrame(leakage_data).sort_values(by='Rejection Driver Severity (%)', ascending=True)
                
                fig4 = px.bar(
                    leakage_df, 
                    x='Rejection Driver Severity (%)', 
                    y='Feature', 
                    orientation='h',
                    title="Top Drivers Causing Rejections (Relative Severity)",
                    color='Rejection Driver Severity (%)',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("No loans rejected at this threshold. Increase threshold to analyze leakage.")

    # --- Insight 3: Expected Profitability Curve ---
    with tab3:
        st.subheader("4. Expected Portfolio Profitability Curve (Financial Optimization)")
        st.markdown("**Executive Summary:** Proves to the C-Suite exactly what acceptable risk threshold maximizes bottom-line revenue.")
        
        thresholds = np.arange(0.1, 0.95, 0.05)
        expected_profits = []
        
        for t in thresholds:
            # Simulate Profit: 
            # If default <= t (Approved): Good outcome = 10% of loan amount, Bad outcome = -80% of loan amount.
            approved_mask = df['prob_default'] <= t
            probs = df.loc[approved_mask, 'prob_default']
            loans = df.loc[approved_mask, 'loan_amnt']
            
            ev_good = (1 - probs) * (0.10 * loans)
            ev_bad = probs * (-0.80 * loans)
            total_profit = (ev_good + ev_bad).sum()
            expected_profits.append(total_profit)
            
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=thresholds, y=expected_profits, mode='lines+markers', line=dict(color='blue', width=3)))
        
        # Find peak
        max_profit = max(expected_profits)
        best_threshold = thresholds[np.argmax(expected_profits)]
        
        fig3.add_vline(x=best_threshold, line_dash="dash", line_color="green", annotation_text=f"Peak: {best_threshold:.0%}")
        fig3.add_hline(y=0, line_dash="dot", line_color="red")
        
        fig3.update_layout(
            title="Total Expected Portfolio Profitability vs. Risk Tolerance",
            xaxis_title="Maximum Acceptable Default Probability Cutoff",
            yaxis_title="Expected Total Profit ($)"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # --- Insight 5: Demographic Risk Exposure Heatmap ---
    with tab4:
        st.subheader("5. The Demographic Risk Exposure Heatmap (Systemic Risk)")
        st.markdown("**Executive Summary:** Highlights structural portfolio vulnerabilities. Helps risk committees discover non-obvious segments (e.g., highly-paid but short credit history applicants defaulting).")
        
        # Create bins for heatmap
        df_heat = df.copy()
        
        if 'person_emp_length' in df_heat.columns and 'person_income' in df_heat.columns:
            df_heat['Emp_Length_Bins'] = pd.cut(df_heat['person_emp_length'], bins=[-1, 2, 5, 10, 50], labels=['0-2 yrs', '3-5 yrs', '6-10 yrs', '10+ yrs'])
            df_heat['Income_Bins'] = pd.cut(df_heat['person_income'], bins=[-1, 30000, 60000, 100000, 1000000], labels=['<30k', '30k-60k', '60k-100k', '100k+'])
            
            heatmap_data = df_heat.groupby(['Income_Bins', 'Emp_Length_Bins'], observed=True)['prob_default'].mean().reset_index()
            
            fig5 = px.density_heatmap(
                heatmap_data, 
                x="Emp_Length_Bins", 
                y="Income_Bins", 
                z="prob_default",
                histfunc="avg",
                color_continuous_scale="Reds",
                title="Systemic Vulnerability: Employment vs Income Brackets",
                labels={'Emp_Length_Bins': 'Employment Length', 'Income_Bins': 'Income Bracket', 'prob_default': 'Avg Risk (Prob Default)'}
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.error("Required demographic features ('person_emp_length', 'person_income') are missing from the dataset.")


page = st.sidebar.selectbox("Dashboard View", ["Active Decision Engine", "Portfolio Insights"])

if page == "Active Decision Engine":
    # Sidebar - Applicant Details
    st.sidebar.header("Applicant Details")
    input_mode = st.sidebar.radio("Input Method", ["Manual Entry", "Upload JSON"])

    if input_mode == "Manual Entry":
        age = st.sidebar.slider("Age", 18, 100, 30)
        income = st.sidebar.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
        emp_length = st.sidebar.slider("Employment Length (Years)", 0, 50, 5)
        loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=0, value=15000, step=500)
        int_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 25.0, 10.5)
        cred_hist_length = st.sidebar.slider("Credit History Length (Years)", 0, 30, 4)
        home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        prior_default = st.sidebar.selectbox("Prior Default on File", ["No", "Yes"])
        ready_to_assess = st.sidebar.button("Assess Risk", type="primary")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Applicant JSON (.json)", type=["json"])
        if uploaded_file is not None:
            try:
                record = json.load(uploaded_file)
                age = record.get("person_age", 30)
                income = record.get("person_income", 60000)
                emp_length = record.get("person_emp_length", 5)
                loan_amnt = record.get("loan_amnt", 15000)
                int_rate = record.get("loan_int_rate", 10.5)
                cred_hist_length = record.get("cb_person_cred_hist_length", 4)
                
                # Helper to match home ownership strings roughly
                ho_val = record.get("home_ownership", "RENT").upper()
                if ho_val not in ["RENT", "OWN", "MORTGAGE", "OTHER"]: ho_val = "RENT"
                home_ownership = ho_val
                
                prior_default = record.get("prior_default", "No")
                
                st.sidebar.success("JSON loaded successfully! Click Assess Risk.")
                ready_to_assess = st.sidebar.button("Assess Risk", type="primary")
            except Exception as e:
                st.sidebar.error(f"Error parsing JSON: {e}")
                ready_to_assess = False
        else:
            st.sidebar.info("Please upload a .json file to continue.")
            st.sidebar.markdown("**Example JSON format:**")
            st.sidebar.code('{\n  "person_age": 25,\n  "person_income": 40000,\n  "person_emp_length": 2,\n  "loan_amnt": 14000,\n  "loan_int_rate": 14.5,\n  "cb_person_cred_hist_length": 3,\n  "home_ownership": "RENT",\n  "prior_default": "Yes"\n}', language='json')
            ready_to_assess = False

    if ready_to_assess:
        
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
        
        # Filter out 'loan_grade' from the dynamic recommendation logic
        # because the user doesn't input it, so we don't want it to constantly trigger.
        filtered_contributions = {k: v for k, v in feature_contributions.items() if 'grade' not in k.lower()}
        
        sorted_drivers = sorted(filtered_contributions.items(), key=lambda x: x[1], reverse=True)
        top_2_drivers = [item for item in sorted_drivers if item[1] > 0][:2]
        
        # Find top mitigating driver (most negative SHAP value)
        mitigating_drivers = sorted(filtered_contributions.items(), key=lambda x: x[1])
        top_mitigator = mitigating_drivers[0] if len(mitigating_drivers) > 0 and mitigating_drivers[0][1] < 0 else None

        st.markdown("---")
        st.subheader("2. Automated Business Recommendation")
        
       
        if len(top_2_drivers) == 0:
            st.success("PROCEED: No significant dynamic risk drivers identified. Approve under standard protocol.")
        else:
            top_driver = top_2_drivers[0][0].lower()
            
            if 'loan_to_income' in top_driver or 'percent_income' in top_driver:
                st.error("RECOMMENDATION: Counter-offer with a lower principal loan amount to bring the LTI ratio under 35%.")
            if 'int_rate' in top_driver:
                st.warning("RECOMMENDATION: Reject current unsecured terms, but offer a secured loan option to lower the interest burden.")
            if 'person_income' in top_driver:
                st.error("RECOMMENDATION: Income is flagged as insufficient for this loan structure. Request proof of additional income or a co-signer.")
            if 'home_ownership' in top_driver:
                st.warning("RECOMMENDATION: Housing instability flagged. Require a larger down payment or collateral to mitigate flight risk.")
            if 'cred_hist_length' in top_driver:
                st.warning("RECOMMENDATION: Short credit history driving risk. Require secondary credit references (e.g., utility bills).")
            if 'emp_length' in top_driver:
                st.warning("RECOMMENDATION: Short or unstable employment history flagged. Require most recent pay stubs and employer verification.")
            if not any(kw in top_driver for kw in ['loan_to_income', 'percent_income', 'int_rate', 'person_income', 'home_ownership', 'cred_hist_length', 'emp_length']):
                st.error(f"RECOMMENDATION: Elevated risk stemming from `{top_2_drivers[0][0]}`. Escalate to human underwriter for manual review.")

        # 4. Visual Explainability (Local SHAP)
        st.markdown("---")
        st.subheader("3. Why was this decision made?")
        st.markdown("""
        Our AI analyzed this application and calculated a **Probability of Default**. The chart below acts as a **'Balance Scale'** for this decision:
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            🔴 **Red (Risk Drivers):** These factors push the probability **UP**. 
            *Think of these as 'Red Flags' that increase the chance of default.*
            """)
        with col2:
            st.markdown("""
            🔵 **Blue (Safety Drivers):** These factors push the probability **DOWN**. 
            *Think of these as 'Safety Nets' that increase the chance of repayment.*
            """)

        st.markdown("""
        *The **Final Result** (at the top of the chart) is where the balance settled.*
        """)
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(local_shap_vals[0], show=False)
        st.pyplot(fig)
else:
    show_portfolio_insights()
