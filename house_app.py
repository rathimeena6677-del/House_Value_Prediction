import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")
st.title("🏠 House Price Prediction")
st.write("This app predicts median house value using Polynomial Linear Regression.")

# =========================================================
# STEP 1: LOAD THE DATA
# housing.csv must be in the same folder as this app.py
# =========================================================
df = pd.read_csv("housing.csv")

# Fill missing bedroom values with the median (same as the notebook)
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# =========================================================
# STEP 2: PREPARE FEATURES AND TARGET
# "ocean_proximity" is text (e.g. "NEAR OCEAN"), so we convert it into
# separate 0/1 columns using one-hot encoding (get_dummies).
# =========================================================
X = df.drop("median_house_value", axis=1)
X = pd.get_dummies(X, columns=["ocean_proximity"], drop_first=True)
y = df["median_house_value"]

ORIGINAL_NUMERIC_FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]
OCEAN_PROXIMITY_OPTIONS = sorted(df["ocean_proximity"].unique().tolist())

# =========================================================
# STEP 3: SPLIT INTO TRAINING AND TESTING DATA
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# STEP 4: TRAIN THE EVALUATION MODEL (degree = 2)
# Polynomial Features let a Linear Regression model capture curved
# (non-straight-line) relationships. Degree 2 means it also looks at
# squared terms and pairwise interactions between features.
# This model is trained only on the TRAINING data, so we can fairly
# test it on data it has never seen.
# =========================================================
DEGREE = 2
poly = PolynomialFeatures(degree=DEGREE)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

eval_model = LinearRegression()
eval_model.fit(X_train_poly, y_train)

y_pred = eval_model.predict(X_test_poly)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# =========================================================
# STEP 5: TRAIN THE FINAL MODEL FOR PREDICTIONS
# This one is trained on ALL the data (not just 80%), so the app's
# live predictions use everything the model can learn from.
# =========================================================
final_poly = PolynomialFeatures(degree=DEGREE)
X_all_poly = final_poly.fit_transform(X)

final_model = LinearRegression()
final_model.fit(X_all_poly, y)

# =========================================================
# STEP 6: BUILD THE APP LAYOUT (3 tabs)
# =========================================================
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Data Overview", "📈 Model Performance"])

# ---------------------------------------------------------
# TAB 1: Let the user enter house details and predict
# ---------------------------------------------------------
with tab1:
    st.subheader("Enter House Details")

    user_input = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(ORIGINAL_NUMERIC_FEATURES):
        column = col1 if i % 2 == 0 else col2
        default_value = float(df[feature].median())
        user_input[feature] = column.number_input(feature, value=default_value)

    ocean_proximity = st.selectbox("Ocean Proximity", OCEAN_PROXIMITY_OPTIONS)

    if st.button("Predict House Price", type="primary"):
        # Build a one-row DataFrame from the user's inputs
        input_df = pd.DataFrame([user_input])
        input_df["ocean_proximity"] = ocean_proximity

        # One-hot encode it the same way the training data was encoded
        input_encoded = pd.get_dummies(input_df, columns=["ocean_proximity"], drop_first=True)

        # Add any missing dummy columns (e.g. if the chosen category didn't
        # produce a column) and put columns in the same order as training data
        for col in X.columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[X.columns]

        # Apply the same polynomial transformation used in training
        input_poly = final_poly.transform(input_encoded)

        predicted_price = final_model.predict(input_poly)[0]
        st.success(f"Predicted House Price: **${predicted_price:,.2f}**")

# ---------------------------------------------------------
# TAB 2: Show basic info about the dataset
# ---------------------------------------------------------
with tab2:
    st.subheader("First Rows of the Dataset")
    st.dataframe(df.head(20))

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Houses by Ocean Proximity")
    st.bar_chart(df["ocean_proximity"].value_counts())

# ---------------------------------------------------------
# TAB 3: Show how well the model performed
# ---------------------------------------------------------
with tab3:
    st.subheader(f"Evaluation Metrics (Polynomial Degree {DEGREE}, tested on unseen data)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:,.0f}")
    col2.metric("MSE", f"{mse:,.0f}")
    col3.metric("RMSE", f"{rmse:,.0f}")
    col4.metric("R2 Score", f"{r2:.3f}")

    st.subheader("Actual vs Predicted Prices")
    st.write("Points closer to the diagonal red line mean more accurate predictions.")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.4)
    slope, intercept = np.polyfit(y_test, y_pred, 1)
    ax.plot(y_test, slope * y_test + intercept, "r", linewidth=2)
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Actual Prices vs Predicted Prices")
    st.pyplot(fig)

    st.subheader("Comparing Polynomial Degrees")
    st.write(
        "Higher degrees can fit the training data more closely, but may "
        "overfit. This compares R2 score across degrees, using all the data."
    )
    degree_results = []
    for d in range(1, 4):
        poly_d = PolynomialFeatures(degree=d)
        X_poly_d = poly_d.fit_transform(X)
        model_d = LinearRegression()
        model_d.fit(X_poly_d, y)
        r2_d = r2_score(y, model_d.predict(X_poly_d))
        degree_results.append({"Degree": d, "R2 Score": round(r2_d, 4)})
    st.dataframe(pd.DataFrame(degree_results))
