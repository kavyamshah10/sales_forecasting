# ==============================
# IMPORTS
# ==============================
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# IMPORTANT: import from script folder
from script.forecast_2013 import forecast_2013


# ==============================
# LOAD DATA & MODEL
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/test_fe.csv")   # update if using train_fe.csv

@st.cache_resource
def load_model():
    return joblib.load("model/best_model.pkl")


data = load_data()
model = load_model()

st.title("📊 Sales Forecasting App")


# ==============================
# SIDEBAR
# ==============================
option = st.sidebar.radio(
    "Select Option",
    ["📈 Model Evaluation (2012)", "🔮 Predict 2013 Sales"]
)


# ==============================
# OPTION 1: 2012 EVALUATION
# ==============================
if option == "📈 Model Evaluation (2012)":

    st.header("📊 2012 Actual vs Predicted")

    test_data = data[data['Year'] == 2012].copy()

    X_test = test_data.drop(columns=['Weekly_Sales','Date'])
    y_test = test_data['Weekly_Sales']

    y_pred = model.predict(X_test)
    test_data['Predicted'] = y_pred

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("RMSE", f"{rmse:,.2f}")
    col3.metric("R²", f"{r2:.4f}")

    # Graph
    agg_df = test_data.groupby('week')[['Weekly_Sales', 'Predicted']].mean().reset_index()

    fig, ax = plt.subplots()
    ax.plot(agg_df['week'], agg_df['Weekly_Sales'], label="Actual")
    ax.plot(agg_df['week'], agg_df['Predicted'], label="Predicted")

    ax.set_title("2012 Sales Comparison")
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)


# ==============================
# OPTION 2: 2013 FORECAST
# ==============================
elif option == "🔮 Predict 2013 Sales":

    st.header("🔮 Forecast 2013 Sales")

    store = st.selectbox("Select Store", sorted(data['Store'].unique()))
    weeks = st.slider("Weeks", 1, 52, 12)

    if st.button("Run Forecast"):

        forecast_df = forecast_2013(model, data, weeks)

        store_df = forecast_df[forecast_df['Store'] == store]

        st.success("Forecast Generated!")

        # Graph
        fig, ax = plt.subplots()
        ax.plot(store_df['week'], store_df['Weekly_Sales'], label="Predicted")

        ax.set_title(f"Store {store} Forecast (2013)")
        ax.set_xlabel("Week")
        ax.set_ylabel("Sales")
        ax.legend()

        st.pyplot(fig)

        # Table
        st.dataframe(store_df,hide_index=True)

        # Download
        csv = store_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "Download Forecast",
            csv,
            f"store_{store}_forecast.csv",
            "text/csv"
        )