# 🏦 Intelligent Decision Support System (IDSS) for Credit Risk

Welcome to the **Intelligent Decision Support System (IDSS)**. This repository houses an end-to-end, enterprise-grade Machine Learning solution designed to transform how financial institutions evaluate, understand, and mitigate credit risk.

---

## 🛑 1. Problem Statement
The modern banking system faces a critical bottleneck: the tradeoff between **predictive accuracy** and **regulatory transparency**. 
* Advanced Machine Learning models (like deep neural networks and ensemble trees) are highly accurate at predicting loan defaults, but they act as "Black Boxes". 
* Due to strict regulations (e.g., Fair Lending Acts, Equal Credit Opportunity Act), banks cannot legally deploy models that they cannot interpret. If a loan is denied, the bank must be able to explicitly explain *why* it was denied to the applicant and regulators.
* Because of this constraint, many banks are stuck using outdated, highly manual, or overly simplistic logistic regression scoring systems—costing them millions in potential revenue and exposing them to systemic defaults.

## 💡 2. Business Value
This project bridges the gap between mathematically advanced AI and human-readable business logic. 
* **Revenue Recovery:** By safely deploying a high-performance **XGBoost** model paired with an explainability engine, the bank can identify profitable, low-risk loans that older scorecard models would falsely reject.
* **Loss Mitigation:** The IDSS strictly identifies "High Risk" candidates by cross-referencing interacting variables (e.g., high income but short employment combined with a high loan-to-income ratio) that humans might miss.
* **Operational Efficiency:** It reduces the time human underwriters spend reviewing clear-cut cases. The IDSS auto-approves safe profiles and intercepts risky ones with dynamic, data-driven counter-offer recommendations.

## 🚀 3. What This Adds to the Banking System
This project provides a live, interactive **Active Decision Engine**. It does not just output a "Probability of Default" number. Instead, it:
1. **Unpacks the Black Box:** Uses SHAP (SHapley Additive exPlanations) to legally justify every single approval and denial.
2. **Automates Policy:** Automatically reads the AI's logic to suggest human interventions (e.g., *"Counter-offer with a lower principal because the Loan-to-Income ratio is causing the risk"*).
3. **Monitors Systemic Risk:** Provides the C-Suite with a real-time Portfolio Dashboard to simulate credit-tightening policies, optimize profit thresholds, and visualize leakage.

---

## 🏗️ Detailed Project Architecture & The 5 Phases

Unlike standard data science tutorials, this project simulates a complete enterprise MLOps lifecycle divided into 5 distinct phases:

### Phase 1: Exploratory Data Analysis (EDA) & Data Integrity
* **Objective:** Understand the raw financial dataset, uncover statistical distributions, and identify data corruption.
* **Processes Executed:**
  * Conducted deep univariate and bivariate analysis on demographics and historical loan performances.
  * Identified anomalies and outliers (e.g., applicants with impossible age bounds or hyper-inflated incomes).
  * Visualized the class imbalance between "Performing" (Paid) and "Non-Performing" (Defaulted) loans.

### Phase 2: Preprocessing & Feature Engineering
* **Objective:** Transform raw data into a mathematically optimized matrix for the XGBoost algorithm.
* **Processes Executed:**
  * **Missing Value Imputation:** Handled missing employment lengths and interest rates safely to preserve data distributions.
  * **Feature Engineering:** Derived crucial financial calculations that heavily dictate risk. E.g., dynamically calculating the `loan_to_income_ratio` (Leverage) and `emp_length_to_age_ratio`.
  * **Categorical Encoding:** Applied One-Hot Encoding (OHE) to translate text strings (like `Home_Ownership = RENT`) into binary vectors.
  * **Data Splitting:** Segregated the dataset into Train/Test subsets to prevent data leakage during model evaluation.

### Phase 3: Predictive Modeling & Optimization
* **Objective:** Train a robust, non-linear machine learning model capable of capturing complex interacting risk factors.
* **Processes Executed:**
  * **Algorithm Selection:** Selected **XGBoost** (Extreme Gradient Boosting) for its peerless performance on tabular financial data.
  * **Hyperparameter Tuning:** Optimized tree depth, learning rate, and class weighting (`scale_pos_weight`) to force the algorithm to care about identifying minority default cases.
  * **Model Serialization:** Serialized the optimized model into `best_credit_risk_model.joblib` to detach it from the training environment and prep it for the Streamlit UI.
  * *(Note: We actively debugged a severe deployment issue involving Streamlit Cloud parsing errors (the `[5E-1]` base_score bug), resolving it by strictly pinning `xgboost==1.7.6` in our environment).* 

### Phase 4: Machine Learning Explainability (XAI) & Segmentation
* **Objective:** Break open the XGBoost "Black Box" using SHAP (SHapley Additive exPlanations) to extract business logic.
* **Processes Executed:**
  * **Global SHAP Analysis:** Extracted overall feature importance to prove that `loan_to_income_ratio` and `loan_int_rate` were the dominant systemic drivers of portfolio default.
  * **Customer Persona Profiling:** Clustered the applicants based on their predicted probabilities into High, Medium, and Low risk tiers. Uncovered clear personas such as "The Over-leveraged Renter" and "The Stable Homeowner".
  * **Local SHAP Generation:** Built the code to extract individual decision waterfalls, simulating the localized output needed for Fair Lending Act compliance.

### Phase 5: The Streamlit Prototype (Active Decision Engine)
* **Objective:** Deploy the model into an intuitive UI/UX for both Underwriters and C-Suite Risk Managers.
* **Processes Executed:**
  * **Dynamic Business Rules Engine:** Built a logic flow that reads individual SHAP scores and dynamically suggests specific counter-offers (e.g., if income is the top SHAP risk driver, the UI automatically recommends requesting a co-signer).
  * **Bias Mitigation:** Actively filtered out static/biased features (like `loan_grade`) from the dynamic recommendation system to prevent the engine from repeating useless warnings.
  * **JSON Batch Assessment:** Constructed a backend file-uploader allowing risk analysts to evaluate structured JSON applicant data rather than typing inputs manually.

---

## 🖥️ The Streamlit Application Visuals

### A. Active Decision Engine (For Underwriters)
The Underwriter view processes individuals (via manual slider entry or JSON) and utilizes the **Dynamic Rule Engine** to reject/approve with clear, SHAP-derived justifications.

![Active Decision Engine](images/active_decision_engine.jpg)
*Figure: The individual Underwriter View showing an extreme risk (85.47% Probability of Default) and a tailored policy recommendation.*

![SHAP Local Explainability](images/shap_waterfall.jpg)
*Figure: The SHAP Waterfall "Balance Scale" plot. The red bars prove why the applicant's probability was driven up, satisfying regulatory transparency laws.*

### B. Portfolio Insights & Strategic Dashboard (For Executives)
A macro-level intelligence suite utilizing `Plotly` to help Risk Committees optimize bank-wide policies.

![Portfolio Insights Overview](images/portfolio_overview.jpg)
*Figure: Top-level KPIs tracking total exposure ($63M) and average portfolio-wide default probabilities across the entire tested dataset.*

#### 1. The Risk vs. Return Scatter Matrix
Plots the foundational law of finance: Risk vs. expected Return. Helps executives identify the "Gold Mine" (high interest, low risk) vs. the "Danger Zone".
![Risk vs Return](images/risk_vs_return.jpg)

#### 2. The Dynamic Approval Funnel & Leakage Tracker
An interactive simulator. Risk managers can drag the "acceptable risk" threshold (e.g., 30%) and instantly see how many millions of dollars the bank will approve or reject. The "Leakage" chart reveals the systemic reasons *why* good revenue was lost.
![Dynamic Funnel](images/dynamic_funnel.jpg)

#### 3. Expected Portfolio Profitability Curve
Calculates simulated Expected Value (EV) curves to prove to the Board precisely which risk threshold (e.g., 10%) mathematically maximizes bottom-line portfolio revenue.
![Profitability Curve](images/profit_optimization.jpg)

#### 4. Demographic Risk Exposure Heatmap
Exposes hidden systemic risks, such as highly paid applicants who still exhibit extreme default probabilities due to structural factors like poor credit history or short employment lengths.
![Demographic Exposure Heatmap](images/demographic_heatmap.jpg)

---

## ⚙️ Setup & Installation

To run the IDSS locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd IDSS-Credit-Risk
   ```

2. **Install the required packages:**
   *(⚠️ CRITICAL: Ensure you are using `xgboost==1.7.6` and `shap==0.44.0`. Newer versions of XGBoost break the local Streamlit environment due to base_score array formatting `[5E-1]`)*
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 📜 Technology Stack
* **Machine Learning:** `scikit-learn`, `xgboost` 1.7.6
* **Explainable AI (XAI):** `shap` 0.44.0
* **Data Manipulation:** `pandas`, `numpy`
* **Dashboard & UI:** `streamlit`, `plotly`, `matplotlib`
