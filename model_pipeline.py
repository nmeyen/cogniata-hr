import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import shap
import pickle

# -----------------------------
# 1. DATA GENERATION
# -----------------------------

def simulate_time_to_hire_data(n=100, start_date="2023-01-01"):
    np.random.seed(42)
    random.seed(42)
    start = datetime.strptime(start_date, "%Y-%m-%d")

    roles = ["Software Engineer", "Data Analyst", "Project Manager", 
             "Marketing Manager", "HR Generalist", "Business Analyst", 
             "DevOps Engineer", "UI/UX Designer", "Sales Associate", 
             "Product Manager"]
    departments = ["Engineering", "Data", "Operations", "Marketing", 
                   "Human Resources", "Strategy", "Design", "Sales", 
                   "Product"]
    seniorities = ["Entry", "Mid", "Senior"]
    sources = ["LinkedIn", "Indeed", "Referral", "Job Fair", "Company Website"]

    data = []
    for i in range(n):
        job_id = i + 1
        role = random.choice(roles)
        department = random.choice(departments)
        seniority = random.choice(seniorities)

        posting_offset = np.random.randint(0, 180)
        posting_date = start + timedelta(days=int(posting_offset))

        time_to_fill = np.random.randint(15, 60)  # 15-60 days
        hire_date = posting_date + timedelta(days=time_to_fill)

        recruit_source = random.choice(sources)
        market_conditions_score = np.random.uniform(0.5, 1.0)

        data.append({
            "Job_ID": job_id,
            "Role": role,
            "Department": department,
            "Seniority": seniority,
            "Posting_Date": posting_date,
            "Hire_Date": hire_date,
            "Time_to_Fill": time_to_fill,
            "Recruit_Source": recruit_source,
            "Market_Conditions_Score": round(market_conditions_score, 2)
        })

    return pd.DataFrame(data)

def simulate_candidate_data(n=300):
    np.random.seed(42)
    random.seed(42)

    education_levels = ["Associate", "Bachelor", "Master", "PhD"]
    data = []
    for i in range(n):
        candidate_id = f"C{i+1:04d}"
        job_id = np.random.randint(1, 101)

        education_level = random.choice(education_levels)
        years_exp = np.random.randint(0, 10)
        interview_score = np.random.randint(60, 100)
        cultural_fit_score = np.random.randint(60, 100)

        # Probability of being hired
        hired_prob = (interview_score + cultural_fit_score) / 200
        hired = "Yes" if np.random.rand() < hired_prob else "No"

        performance_after_6_months = np.random.randint(60, 100) if hired == "Yes" else None

        data.append({
            "Candidate_ID": candidate_id,
            "Job_ID": job_id,
            "Education_Level": education_level,
            "Years_of_Experience": years_exp,
            "Interview_Score": interview_score,
            "Cultural_Fit_Score": cultural_fit_score,
            "Performance_After_6_Months": performance_after_6_months,
            "Hired": hired
        })

    return pd.DataFrame(data)

# -----------------------------
# 2. MEAN ENCODING HELPER
# -----------------------------
def mean_encode(df, col_to_encode, target_col):
    """
    Applies mean encoding for a given categorical column based on the target column.
    Mutates the DataFrame in place, adding a new column: <col_to_encode>_mean_enc.
    """
    mean_map = df.groupby(col_to_encode)[target_col].mean()
    new_col = f"{col_to_encode}_mean_enc"
    df[new_col] = df[col_to_encode].map(mean_map)
    return df

# -----------------------------
# 3. MODEL TRAINING & SHAP
# -----------------------------
def train_time_to_fill_model(df_time):
    """
    Trains a regression model to predict Time_to_Fill using mean encoding.
    Returns the fitted model, train/test data, and SHAP values.
    """
    df_time["Posting_Date"] = pd.to_datetime(df_time["Posting_Date"])
    df_time["Hire_Date"] = pd.to_datetime(df_time["Hire_Date"])

    # Mean-encode categorical variables
    cat_cols = ["Role", "Department", "Seniority", "Recruit_Source"]
    for col in cat_cols:
        mean_encode(df_time, col, "Time_to_Fill")
    
    # Drop original categorical columns
    df_time.drop(columns=cat_cols, inplace=True)

    # Prepare features
    X = df_time.drop(["Job_ID", "Posting_Date", "Hire_Date", "Time_to_Fill"], axis=1)
    y = df_time["Time_to_Fill"]

    # Convert any remaining non-numeric columns if needed
    X = X.select_dtypes(include=[np.number])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Compute SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values_test = explainer(X_test)

    return model, X_train, X_test, y_train, y_test, shap_values_test

def train_candidate_success_model(df_candidates):
    """
    Trains a regression model to predict Performance_After_6_Months using mean encoding.
    Only uses rows where Hired='Yes' and Performance_After_6_Months is not null.
    Returns the fitted model, train/test data, and SHAP values.
    """
    df_succ = df_candidates[
        (df_candidates["Hired"] == "Yes") &
        (df_candidates["Performance_After_6_Months"].notna())
    ].copy()

    # If there are no records, return None placeholders
    if df_succ.empty:
        return None, None, None, None, None, None

    # Mean-encode categorical variables
    # You could consider columns like "Education_Level" for encoding
    mean_encode(df_succ, "Education_Level", "Performance_After_6_Months")

    # Drop original categorical columns (excluding "Hired" as it's filtered but you can drop if you prefer)
    df_succ.drop(columns=["Education_Level", "Hired"], inplace=True)

    # Prepare features
    X = df_succ.drop(["Candidate_ID", "Job_ID", "Performance_After_6_Months"], axis=1)
    y = df_succ["Performance_After_6_Months"]

    # Convert to numeric if needed
    X = X.select_dtypes(include=[np.number])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Compute SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values_test = explainer(X_test)

    return model, X_train, X_test, y_train, y_test, shap_values_test

# -----------------------------
# 4. MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    # Generate data
    df_time_to_hire = simulate_time_to_hire_data(n=100)
    df_candidates = simulate_candidate_data(n=300)

    # Train Time-to-Fill model
    (time_to_fill_model,
     X_train_t,
     X_test_t,
     y_train_t,
     y_test_t,
     shap_vals_time_test) = train_time_to_fill_model(df_time_to_hire)

    # Train Candidate Success model
    (candidate_success_model,
     X_train_s,
     X_test_s,
     y_train_s,
     y_test_s,
     shap_vals_succ_test) = train_candidate_success_model(df_candidates)

    # Save everything needed for the Streamlit app
    with open("data_and_models.pkl", "wb") as f:
        pickle.dump({
            "df_time_to_hire": df_time_to_hire,
            "df_candidates": df_candidates,
            "time_to_fill_model": time_to_fill_model,
            "X_test_t": X_test_t,
            "y_test_t": y_test_t,
            "shap_vals_time_test": shap_vals_time_test,
            "candidate_success_model": candidate_success_model,
            "X_test_s": X_test_s,
            "y_test_s": y_test_s,
            "shap_vals_succ_test": shap_vals_succ_test
        }, f)
