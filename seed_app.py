import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Seed Type Prediction", page_icon="🌾", layout="wide")
st.title("🌾 Seed Type Prediction")
st.write("This app predicts the type of wheat seed (Kama, Rosa, or Canadian) using Logistic Regression.")

# =========================================================
# STEP 1: LOAD THE DATA
# seeds_dataset.txt must be in the same folder as this app.py
# The file has no header row and is separated by whitespace,
# so we give the column names ourselves.
# =========================================================
COLUMN_NAMES = [
    "Area",
    "Perimeter",
    "Compactness",
    "Kernel_Length",
    "Kernel_Width",
    "Asymmetry_Coefficient",
    "Kernel_Groove_Length",
    "Class",
]

df = pd.read_csv(
    "seeds_dataset.txt",
    sep=r"\s+",
    header=None,
    names=COLUMN_NAMES,
    engine="python",
)

# These are the 7 measurement columns used to predict the seed type.
# "Class" (1, 2, or 3) is the target we want to predict, so it is
# NOT included in this list.
FEATURES = [
    "Area", "Perimeter", "Compactness", "Kernel_Length",
    "Kernel_Width", "Asymmetry_Coefficient", "Kernel_Groove_Length",
]

# Map the numeric class to a readable seed name (same as the notebook)
CLASS_NAMES = {1: "Kama", 2: "Rosa", 3: "Canadian"}

X = df[FEATURES]
y = df["Class"]

# =========================================================
# STEP 2: SPLIT INTO TRAINING AND TESTING DATA
# 80% of the data is used to train the model.
# 20% is kept aside to test how well the model performs on new data.
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# STEP 3: TRAIN THE MODEL
# (No scaling is used here, matching the original notebook.)
# =========================================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================================================
# STEP 4: EVALUATE THE MODEL
# Check how accurate the model is on the test data it has never seen.
# =========================================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# =========================================================
# STEP 5: BUILD THE APP LAYOUT (3 tabs)
# =========================================================
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Data Overview", "📈 Model Performance"])

# ---------------------------------------------------------
# TAB 1: Let the user enter seed measurements and predict
# ---------------------------------------------------------
with tab1:
    st.subheader("Enter Seed Measurements")
    st.write("Adjust each value, then click Predict.")

    user_input = {}
    col1, col2 = st.columns(2)

    # Create one number input per feature.
    # We split them into two columns so the page isn't one long list.
    for i, feature in enumerate(FEATURES):
        column = col1 if i % 2 == 0 else col2
        default_value = float(df[feature].mean())  # start at the average value
        user_input[feature] = column.number_input(feature, value=default_value)

    if st.button("Predict Seed Type", type="primary"):
        # Put the user's inputs into a single-row table in the correct column order
        input_df = pd.DataFrame([user_input])[FEATURES]

        # Predict the class number, then convert it to a readable name
        prediction_number = model.predict(input_df)[0]
        prediction_name = CLASS_NAMES.get(prediction_number, str(prediction_number))

        st.success(f"Predicted Seed Type: **{prediction_name}**")

        # Get probability for each class, so we can show a confidence chart
        probabilities = model.predict_proba(input_df)[0]
        proba_df = pd.DataFrame({
            "Seed Type": [CLASS_NAMES.get(c, str(c)) for c in model.classes_],
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        st.write("Confidence for each seed type:")
        st.bar_chart(proba_df.set_index("Seed Type"))

# ---------------------------------------------------------
# TAB 2: Show basic info about the dataset
# ---------------------------------------------------------
with tab2:
    st.subheader("First Rows of the Dataset")
    st.dataframe(df.head(20))

    st.subheader("How Many Seeds of Each Type")
    df_display = df.copy()
    df_display["Seed Type"] = df_display["Class"].map(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Seed Type", data=df_display, ax=ax)
    st.pyplot(fig)

    st.subheader("Summary Statistics")
    st.dataframe(df[FEATURES].describe())

# ---------------------------------------------------------
# TAB 3: Show how well the model performed
# ---------------------------------------------------------
with tab3:
    st.subheader("Overall Accuracy")
    st.metric("Accuracy on Test Data", f"{accuracy * 100:.2f}%")

    st.subheader("Confusion Matrix")
    st.write(
        "This shows how many seeds of each type were predicted correctly "
        "(diagonal) versus mistaken for another type (off-diagonal)."
    )
    labels = [CLASS_NAMES.get(c, str(c)) for c in model.classes_]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Detailed Report (Precision, Recall, F1-score)")
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Feature Importance")
    st.write("How much each measurement influences the prediction (from the model's coefficients).")
    coef = pd.Series(model.coef_[0], index=FEATURES)
    fig, ax = plt.subplots(figsize=(8, 5))
    coef.sort_values().plot(kind="barh", ax=ax)
    plt.xlabel("Coefficient Value")
    st.pyplot(fig)
