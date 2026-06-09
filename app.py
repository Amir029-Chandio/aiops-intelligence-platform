import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AIOps Dashboard", layout="wide")

st.title("🚀 AI-Powered AIOps Monitoring System")

# Dataset
np.random.seed(42)
n = 2000

df = pd.DataFrame({
    "CPU_Usage": np.random.randint(10,100,n),
    "Memory_Usage": np.random.randint(20,100,n),
    "Latency_ms": np.random.randint(1,300,n),
    "Packet_Loss": np.random.uniform(0,10,n),
    "Bandwidth_Usage": np.random.randint(100,1000,n)
})

df["Incident"] = (
    (df["CPU_Usage"] > 85) |
    (df["Memory_Usage"] > 90) |
    (df["Latency_ms"] > 220) |
    (df["Packet_Loss"] > 5)
).astype(int)

# Model
X = df.drop("Incident", axis=1)
y = df["Incident"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

st.metric("Total Logs", len(df))
st.metric("Incidents", int(df["Incident"].sum()))

cpu = st.slider("CPU", 0, 100, 50)
mem = st.slider("Memory", 0, 100, 50)
lat = st.slider("Latency", 0, 300, 100)
packet = st.slider("Packet Loss", 0.0, 10.0, 1.0)
bw = st.slider("Bandwidth", 100, 1000, 500)

if st.button("Predict"):
    pred = model.predict([[cpu, mem, lat, packet, bw]])[0]
    if pred == 1:
        st.error("High Incident Risk ⚠")
    else:
        st.success("System Healthy ✅")