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


def _format_value_for_prompt(value):
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.4f}" if abs(float(value)) < 100 else f"{float(value):,.2f}"
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _build_xai_prompt(customer_data, prediction_prob, risk_tier, top_features, target_definition):
    customer_profile_text = "\n".join(f"- {key}: {_format_value_for_prompt(value)}" for key, value in customer_data.items())
    driver_lines = []

    for feature in top_features:
        if isinstance(feature, dict):
            name = feature.get("feature", feature.get("name", "Unknown"))
            value = feature.get("value", feature.get("feature_value", "N/A"))
            impact = feature.get("impact", feature.get("direction", feature.get("contribution", "N/A")))
            driver_lines.append(f"- {name}: value={_format_value_for_prompt(value)}, impact={_format_value_for_prompt(impact)}")
        elif isinstance(feature, (list, tuple)) and len(feature) >= 3:
            driver_lines.append(f"- {feature[0]}: value={_format_value_for_prompt(feature[1])}, impact={_format_value_for_prompt(feature[2])}")
        else:
            driver_lines.append(f"- {_format_value_for_prompt(feature)}")

    driver_text = "\n".join(driver_lines) if driver_lines else "- No top features supplied."

    return f"""You are an XAI (Explainable AI) engine for a Predictive Business System. Your job is to translate mathematical model outputs into clear, actionable business intelligence.

PREDICTION DATA:
- Probability of event: {prediction_prob:.2%} ({risk_tier} Risk)
- Definition: {target_definition}

CUSTOMER/APPLICANT PROFILE:
{customer_profile_text}

TOP MODEL DRIVERS (Mathematical feature contributions):
{driver_text}

STRICT INSTRUCTIONS:
Based ONLY on the data provided above, write a business-focused explanation using EXACTLY these four section headings (each preceded by ###):

### Why This Score
Explain in plain English why this profile received this specific probability. Reference 2-3 explicit data points from their profile. Do not invent data.

### Key Risk Drivers
Translate the 'Top Model Drivers' into business terms. Explain what is pushing the risk up or down and why it matters.

### Business Impact
Assess what this specific prediction means for the business operations or bottom line.

### Recommended Actions
Provide exactly 3 specific, highly personalised actions the business should take for this profile to mitigate risk or retain value.

Constraints:
- Use only the data above.
- Do not mention that you are an AI model.
- Do not add extra sections.
- Do not use bullet points unless they are inside the requested sections.
- Keep the tone professional, plain-English, and business-focused."""


def _get_google_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        try:
            return st.secrets["google_api_key"]
        except Exception:
            return st.session_state.get("GOOGLE_API_KEY")


def _build_recommendation_prompt(customer_data, prediction_prob, risk_tier, top_features, target_definition):
    customer_profile_text = "\n".join(f"- {key}: {_format_value_for_prompt(value)}" for key, value in customer_data.items())
    driver_lines = []

    for feature in top_features:
        if isinstance(feature, dict):
            name = feature.get("feature", feature.get("name", "Unknown"))
            value = feature.get("value", feature.get("feature_value", "N/A"))
            impact = feature.get("impact", feature.get("direction", feature.get("contribution", "N/A")))
            driver_lines.append(f"- {name}: value={_format_value_for_prompt(value)}, impact={_format_value_for_prompt(impact)}")
        elif isinstance(feature, (list, tuple)) and len(feature) >= 3:
            driver_lines.append(f"- {feature[0]}: value={_format_value_for_prompt(feature[1])}, impact={_format_value_for_prompt(feature[2])}")
        else:
            driver_lines.append(f"- {_format_value_for_prompt(feature)}")

    driver_text = "\n".join(driver_lines) if driver_lines else "- No top features supplied."

    return f"""You are an expert credit risk business analyst for a Predictive Business System.

PREDICTION DATA:
- Probability of event: {prediction_prob:.2%} ({risk_tier} Risk)
- Definition: {target_definition}

CUSTOMER/APPLICANT PROFILE:
{customer_profile_text}

TOP MODEL DRIVERS (Mathematical feature contributions):
{driver_text}

STRICT INSTRUCTIONS:
Based ONLY on the data above, produce a concise, business-focused recommendation.

Use exactly these three sections:

### Risk Interpretation
Explain in one short paragraph what the score means for this applicant.

### Recommendation
State the primary business decision or next step.

### Actions
Provide exactly 3 concrete actions the business should take right now.

Constraints:
- Do not invent data.
- Do not mention that you are an AI model.
- Keep it specific to the applicant.
- Keep the tone professional and plain-English."""


def _render_summary_strip(prediction_prob, risk_tier, top_features):
    top_feature_name = "N/A"
    if top_features:
        first = top_features[0]
        if isinstance(first, dict):
            top_feature_name = first.get("feature", first.get("name", "N/A"))
        elif isinstance(first, (list, tuple)):
            top_feature_name = first[0] if len(first) > 0 else "N/A"
        else:
            top_feature_name = str(first)

    s1, s2, s3 = st.columns(3)
    s1.metric("Probability", f"{prediction_prob:.2%}")
    s2.metric("Risk Tier", risk_tier)
    s3.metric("Top Driver", top_feature_name)


def display_ai_recommendation(customer_data, prediction_prob, risk_tier, top_features, target_definition="Customer churn / credit default event"):
    st.subheader("🧭 AI Recommendation")
    st.markdown("This section uses Gemini to turn the prediction into a grounded business recommendation based only on the provided inputs.")
    _render_summary_strip(prediction_prob, risk_tier, top_features)

    with st.expander("Show recommendation inputs", expanded=False):
        st.json({
            "customer_data": customer_data,
            "prediction_prob": prediction_prob,
            "risk_tier": risk_tier,
            "top_features": top_features,
        })

    if st.button("✨ Generate AI Recommendation", key=f"generate_ai_recommendation_{target_definition}"):
        with st.spinner("Analysing prediction and generating business recommendation..."):
            try:
                from google import genai

                api_key = _get_google_api_key()
                if not api_key:
                    st.error("Google API key not found. Add GOOGLE_API_KEY to Streamlit secrets or environment variables.")
                    return

                prompt = _build_recommendation_prompt(customer_data, prediction_prob, risk_tier, top_features, target_definition)
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )

                answer = getattr(response, "text", None) or str(response)

                st.markdown("### Recommendation Output")
                st.markdown(
                    f"""
                    <div style="padding: 1rem 1.1rem; border-radius: 0.8rem; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);">
                    {answer.replace(chr(10), '<br>')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as exc:
                st.error(f"Could not generate AI recommendation: {exc}")
                st.info("Fallback: the recommendation component is ready, but the Gemini request failed. Check your API key, internet access, and google-genai installation.")


def display_xai_explanation(customer_data, prediction_prob, risk_tier, top_features, target_definition="Customer churn / credit default event"):
    st.subheader("✨ AI Explanation")
    st.markdown("This component asks Gemini to convert model outputs into a plain-English business explanation, but only from the data you provide.")
    _render_summary_strip(prediction_prob, risk_tier, top_features)

    with st.expander("Show input data", expanded=False):
        st.json({
            "customer_data": customer_data,
            "prediction_prob": prediction_prob,
            "risk_tier": risk_tier,
            "top_features": top_features,
        })

    if st.button("✨ Generate AI Explanation", key=f"generate_xai_explanation_{target_definition}"):
        with st.spinner("Analysing prediction and generating business insights..."):
            try:
                from google import genai

                api_key = _get_google_api_key()

                if not api_key:
                    st.error("Google API key not found. Add GOOGLE_API_KEY to Streamlit secrets or environment variables.")
                    return

                prompt = _build_xai_prompt(customer_data, prediction_prob, risk_tier, top_features, target_definition)

                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )

                answer = getattr(response, "text", None) or str(response)

                st.markdown("### AI Explanation")
                st.container(border=True)
                st.markdown(
                    f"""
                    <div style="padding: 1rem 1.1rem; border-radius: 0.8rem; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);">
                    {answer.replace(chr(10), '<br>')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.info("The response above is constrained to the provided customer profile, prediction score, and top model drivers.")

            except Exception as exc:
                st.error(f"Could not generate AI explanation: {exc}")
                st.info("Fallback: the explanation component is ready, but the Gemini request failed. Check your API key, internet access, and google-genai installation.")


def run_xai_demo():
    st.divider()
    st.header("Demo: AI Explanation Component")
    st.caption("This demo uses synthetic churn-style inputs so you can test the UI immediately.")

    demo_customer_data = {
        "customer_id": "CUST-10421",
        "age": 38,
        "monthly_charges": 89.50,
        "tenure_months": 7,
        "contract_type": "Month-to-month",
        "support_calls_last_90d": 6,
        "payment_method": "Electronic check",
        "region": "West",
        "account_balance": 143.22,
    }

    demo_top_features = [
        {"feature": "support_calls_last_90d", "value": 6, "impact": "increases risk"},
        {"feature": "tenure_months", "value": 7, "impact": "increases risk"},
        {"feature": "monthly_charges", "value": 89.50, "impact": "increases risk"},
        {"feature": "account_balance", "value": 143.22, "impact": "decreases risk"},
    ]

    display_xai_explanation(
        customer_data=demo_customer_data,
        prediction_prob=0.846,
        risk_tier="High",
        top_features=demo_top_features,
        target_definition="Probability that this customer will churn in the next 30 days",
    )


def build_credit_risk_xai_payload(applicant_df, prediction_prob, feature_contributions):
    top_positive = sorted(feature_contributions.items(), key=lambda item: item[1], reverse=True)
    top_negative = sorted(feature_contributions.items(), key=lambda item: item[1])

    top_features = []
    for feature_name, contribution in top_positive[:3]:
        top_features.append({
            "feature": feature_name,
            "value": applicant_df.iloc[0].get(feature_name, "N/A"),
            "impact": f"increases risk by {contribution:.4f}",
        })

    if top_negative:
        feature_name, contribution = top_negative[0]
        top_features.append({
            "feature": feature_name,
            "value": applicant_df.iloc[0].get(feature_name, "N/A"),
            "impact": f"decreases risk by {abs(contribution):.4f}",
        })

    customer_data = applicant_df.iloc[0].to_dict()
    return customer_data, top_features


def render_interactive_shap_waterfall(feature_contributions, base_value, title="Interactive SHAP Waterfall"):
    ranked_items = sorted(feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
    top_items = ranked_items[:10]
    other_sum = sum(value for _, value in ranked_items[10:]) if len(ranked_items) > 10 else 0.0

    labels = [name for name, _ in top_items]
    values = [value for _, value in top_items]

    if abs(other_sum) > 1e-9:
        labels.append("Other features")
        values.append(other_sum)

    labels = ["Baseline"] + labels + ["Final"]
    measures = ["absolute"] + ["relative"] * len(values) + ["total"]
    y_values = [base_value] + values + [None]

    fig = go.Figure(
        go.Waterfall(
            name="SHAP Contribution",
            orientation="v",
            measure=measures,
            x=labels,
            y=y_values,
            connector={"line": {"color": "rgba(200,200,200,0.35)", "width": 1.2}},
            increasing={"marker": {"color": "#ff4d4d"}},
            decreasing={"marker": {"color": "#2b6cb0"}},
            totals={"marker": {"color": "#f5f5f5", "line": {"color": "#cfcfcf", "width": 1}}},
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=560,
        margin=dict(l=30, r=30, t=80, b=45),
        xaxis_title="Model Drivers",
        yaxis_title="Contribution to Model Output",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )

    fig.update_xaxes(tickangle=-25, automargin=True)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")

    st.plotly_chart(fig, use_container_width=True)


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
        recommendation_customer_data, recommendation_top_features = build_credit_risk_xai_payload(applicant_df, prob_default, feature_contributions)
        display_ai_recommendation(
            customer_data=recommendation_customer_data,
            prediction_prob=prob_default,
            risk_tier=tier_text,
            top_features=recommendation_top_features,
            target_definition="Probability that this applicant will default on the loan",
        )

        # 4. Visual Explainability (Interactive SHAP)
        st.markdown("---")
        st.subheader("3. Why was this decision made?")
        st.markdown("""
        Our AI analyzed this application and calculated a **Probability of Default**. The interactive waterfall below shows how the model moved from its baseline view to this specific decision.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            🔴 **Red bars** push the prediction **UP** toward higher risk.
            """)
        with col2:
            st.markdown("""
            🔵 **Blue bars** push the prediction **DOWN** toward lower risk.
            """)

        st.markdown("The chart is interactive: hover to inspect each driver, zoom the view, and compare the strongest positive and negative factors.")
        plot_col, table_col = st.columns([1.6, 1])
        with plot_col:
            render_interactive_shap_waterfall(
                feature_contributions=filtered_contributions,
                base_value=float(local_shap_vals.base_values[0]) if np.ndim(local_shap_vals.base_values) else float(local_shap_vals.base_values),
                title="Interactive SHAP Waterfall: Applicant Risk Drivers",
            )
        with table_col:
            st.markdown("##### Ranked Drivers")
            ranked_driver_rows = []
            for feature_name, contribution in sorted(filtered_contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:8]:
                ranked_driver_rows.append({
                    "Feature": feature_name,
                    "Impact": "Up" if contribution >= 0 else "Down",
                    "SHAP": round(float(contribution), 4),
                })

            ranked_driver_df = pd.DataFrame(ranked_driver_rows)
            st.dataframe(
                ranked_driver_df,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("##### Legend")
            st.markdown(
                """
                <div style="padding: 0.8rem 0.9rem; border-radius: 0.75rem; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);">
                <div><span style="color:#ff4d4d; font-weight:700;">●</span> Red = pushes risk up</div>
                <div><span style="color:#2b6cb0; font-weight:700;">●</span> Blue = pushes risk down</div>
                <div><span style="color:#f5f5f5; font-weight:700;">●</span> Baseline = model starting point</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()
        st.subheader("4. AI Business Explanation")
        st.caption("This explanation is based on the applicant’s actual assessed values and the strongest positive/negative SHAP drivers.")
        real_customer_data, real_top_features = build_credit_risk_xai_payload(applicant_df, prob_default, feature_contributions)
        display_xai_explanation(
            customer_data=real_customer_data,
            prediction_prob=prob_default,
            risk_tier=tier_text,
            top_features=real_top_features,
            target_definition="Probability that this applicant will default on the loan",
        )

else:
    show_portfolio_insights()

if page == "Active Decision Engine":
    st.divider()
    st.subheader("Independent XAI Demo")
    st.caption("Use this section to test the Gemini-powered explanation component without running a loan assessment first.")
    run_xai_demo()
