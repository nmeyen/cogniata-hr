import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

def st_shap(plot, height=300):
    """Helper to display a SHAP plot (force or otherwise) in Streamlit."""
    # This injects SHAP's minimal JavaScript plus the plot's HTML
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def main():
    # Load the logo image
    logo_path = "cog-logo.png"

    # Display the logo in the main area or the sidebar
    # Option A: in the main area
    st.image(logo_path, width=150)  # width can be changed as needed
    st.title("HR - Time to Fill and Candidate Success Visualization App")

    # Load precomputed data and models
    with open("data_and_models.pkl", "rb") as f:
        data_bundle = pickle.load(f)

    df_time = data_bundle["df_time_to_hire"]
    df_candidates = data_bundle["df_candidates"]

    time_model = data_bundle["time_to_fill_model"]
    X_test_t = data_bundle["X_test_t"]
    y_test_t = data_bundle["y_test_t"]
    shap_vals_time_test = data_bundle["shap_vals_time_test"]

    cand_model = data_bundle["candidate_success_model"]
    X_test_s = data_bundle["X_test_s"]
    y_test_s = data_bundle["y_test_s"]
    shap_vals_succ_test = data_bundle["shap_vals_succ_test"]

    st.subheader("Dataset Overviews")
    if st.checkbox("Show Time-to-Hire Data", value=False):
        st.dataframe(df_time.head(10))

    if st.checkbox("Show Candidate Data", value=False):
        st.dataframe(df_candidates.head(10))

    # -----------------------------
    # 1. TIME-TO-FILL VISUALIZATIONS
    # -----------------------------
    st.header("Time-to-Fill Model Insights")

    if time_model is not None:
        st.write("### Feature Correlation Heatmap (Time-to-Hire Encoded)")
        # Create a correlation matrix for the encoded features
        # Re-encode if needed to get the training features
        # (Alternatively, you could load from the pipeline.)
        # Let's assume we can just correlate X_test_t:
        corr_matrix_t = X_test_t.corr()
        fig_corr_t, ax_corr_t = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr_matrix_t, annot=False, cmap="coolwarm", ax=ax_corr_t)
        st.pyplot(fig_corr_t)

        st.write("### SHAP Summary Plot (Time-to-Fill)")
        shap.plots.beeswarm(shap_vals_time_test, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        st.write("### SHAP Bar Plot (Top Features for Time-to-Fill)")
        shap.plots.bar(shap_vals_time_test, max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        st.write("### Force Plot for a Single Row")
        row_index_t = st.slider("Select a row index in X_test for Time-to-Fill", 0, len(X_test_t)-1, 0)
        single_shap_t = shap_vals_time_test[row_index_t]
        st_html_t = shap.plots.force(single_shap_t, matplotlib=False,show=False)
        #st.components.v1.html(st_html_t.html(), height=300)
        st_shap(st_html_t, height=300)

    # -----------------------------
    # 2. CANDIDATE SUCCESS VISUALIZATIONS
    # -----------------------------
    st.header("Candidate Success Model Insights")

    if cand_model is not None:
        st.write("### SHAP Summary Plot (Candidate Success)")
        shap.plots.beeswarm(shap_vals_succ_test, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        st.write("### SHAP Bar Plot (Top Features for Candidate Success)")
        shap.plots.bar(shap_vals_succ_test, max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        st.write("### Force Plot for a Single Row")
        row_index_s = st.slider("Select a row index in X_test for Candidate Success", 0, len(X_test_s)-1, 0)
        single_shap_s = shap_vals_succ_test[row_index_s]
        st_html_s = shap.plots.force(single_shap_s, matplotlib=False,show=False)
        #st.components.v1.html(st_html_s.html(), height=300)
        st_shap(st_html_s, height=300)
    else:
        st.write("Candidate Success Model not available (check data generation).")

    # -----------------------------
    # 3. Additional Visualizations
    # -----------------------------
    st.header("Additional Visualizations")

    st.write("### Distribution of Time_to_Fill")
    fig_dist_t, ax_dist_t = plt.subplots()
    sns.histplot(df_time["Time_to_Fill"], bins=10, kde=True, ax=ax_dist_t)
    ax_dist_t.set_xlabel("Days")
    ax_dist_t.set_ylabel("Count")
    st.pyplot(fig_dist_t)

    # If you want to show actual vs. predicted for time-to-fill:
    if time_model is not None:
        y_pred_t = time_model.predict(X_test_t)
        fig_scatter_t, ax_scatter_t = plt.subplots()
        ax_scatter_t.scatter(y_test_t, y_pred_t, alpha=0.6)
        ax_scatter_t.set_xlabel("Actual Time_to_Fill")
        ax_scatter_t.set_ylabel("Predicted Time_to_Fill")
        ax_scatter_t.set_title("Actual vs. Predicted Time_to_Fill")
        st.pyplot(fig_scatter_t)

    # Similarly for candidate success
    if cand_model is not None and y_test_s is not None:
        y_pred_s = cand_model.predict(X_test_s)
        fig_scatter_s, ax_scatter_s = plt.subplots()
        ax_scatter_s.scatter(y_test_s, y_pred_s, alpha=0.6, color="orange")
        ax_scatter_s.set_xlabel("Actual Performance Score")
        ax_scatter_s.set_ylabel("Predicted Performance Score")
        ax_scatter_s.set_title("Actual vs. Predicted Candidate Performance")
        st.pyplot(fig_scatter_s)

if __name__ == "__main__":
    main()
