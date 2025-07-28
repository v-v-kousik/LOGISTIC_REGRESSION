import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

# Load Data
train_df = pd.read_csv("Titanic_train.csv")

# Preprocessing
def preprocess_data(df):
    df = df.copy()
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['FamilySize'] = df.get('SibSp', 0) + df.get('Parch', 0) + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    drop_cols = [col for col in ['PassengerId', 'Name', 'Ticket', 'Cabin'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    return df

train_df_processed = preprocess_data(train_df.copy())

# Features
X = train_df_processed.drop('Survived', axis=1)
y = train_df_processed['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numerical_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone']
categorical_features = ['Pclass', 'Sex', 'Embarked']

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
])

model_pipeline.fit(X_train, y_train)

# Evaluate Model
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_pred_proba),
}

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Predictor (Logistic Regression)", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict their survival probability using a Logistic Regression model.")

st.header("Model Performance Metrics (on Test Set)")
cols = st.columns(5)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label.capitalize(), f"{value:.2f}")

st.markdown("---")
st.header("Predict Survival for a New Passenger")

# Input Widgets
pclass_display = st.selectbox("Passenger Class", options=['1st Class', '2nd Class', '3rd Class'], index=2)
pclass = {'1st Class': 1, '2nd Class': 2, '3rd Class': 3}[pclass_display]
sex = st.radio("Sex", options=['Male', 'Female']).lower()
age = st.slider("Age", 0, 80, 30)
sibsp = st.slider("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
parch = st.slider("Parents/Children Aboard (Parch)", 0, 6, 0)
fare = st.number_input("Ticket Fare", 0.0, 500.0, 30.0, step=5.0)
embarked = st.selectbox("Port of Embarkation", ['Southampton', 'Cherbourg', 'Queenstown'], index=0)
embarked_code = {'Southampton': 'S', 'Cherbourg': 'C', 'Queenstown': 'Q'}[embarked]

# Inference
input_data = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked_code
}])

if st.button("Predict Survival"):
    input_data_processed = preprocess_data(input_data.copy())
    proba = model_pipeline.predict_proba(input_data_processed)[0, 1]
    prediction = model_pipeline.predict(input_data_processed)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"**Survived!** 🎉 (Probability: {proba*100:.2f}%)")
    else:
        st.error(f"**Did Not Survive.** 😔 (Probability: {(1-proba)*100:.2f}%)")

    st.markdown(f"**Survival Probability:** `{proba*100:.2f}%`")

    st.subheader("Model Insights:")
    st.info("""
    - **Sex:** Females and children had higher survival chances.
    - **Class:** 1st class passengers were prioritized.
    - **Family Size:** Being alone or having a very large family lowered chances.
    - **Fare & Embarkation:** Often reflected social status and lifeboat access.
    """)

# ROC Curve
st.header("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
st.line_chart(roc_df.set_index('False Positive Rate'))

st.markdown("---")
st.caption("Developed with Streamlit and scikit-learn.")
