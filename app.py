import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
st.title("PUBG Machine Learning Models")

df = pd.read_csv("PUBG.csv")
df.dropna(inplace=True)
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))
df["Result"] = (df["Win"] >= 0.5).astype(int)
model_name = st.selectbox(
    "Select Model",
    [
        "Linear Regression",
        "Logistic Regression",
        "KNN",
        "Naive Bayes"
    ]
)
if st.button("Train Model"):
    if model_name == "Linear Regression":
        X = df.drop("Win", axis=1)
        y = df["Win"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.success("Linear Regression Trained Successfully")
        st.write("R² Score:", r2_score(y_test, y_pred))
        st.write("Sample Predictions")
        st.write(y_pred[:10])
    else:
        X = df.drop(["Win", "Result"], axis=1)
        y = df["Result"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=10000)
        elif model_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.success(model_name + " Trained Successfully")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred))
        st.write("Recall:", recall_score(y_test, y_pred))
        st.write("F1 Score:", f1_score(y_test, y_pred))