import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI AIOps Monitoring System", layout="wide")

st.title("🚀 AI-Powered AIOps Monitoring System")
st.markdown("Real-time Network Monitoring + Incident Prediction + AI Insights")

# ---------------- DATASET ----------------
np.random.seed(42)
n = 2000

df = pd.DataFrame({
    "CPU_Usage": np.random.randint(10, 100, n),
    "Memory_Usage": np.random.randint(20, 100, n),
    "Latency_ms": np.random.randint(1, 300, n),
    "Packet_Loss": np.random.uniform(0, 10, n),
    "Bandwidth_Usage": np.random.randint(100, 1000, n)
})

df["Incident"] = (
    (df["CPU_Usage"] > 85) |
    (df["Memory_Usage"] > 90) |
    (df["Latency_ms"] > 220) |
    (df["Packet_Loss"] > 5)
).astype(int)

# ---------------- MODEL ----------------
X = df.drop("Incident", axis=1)
y = df["Incident"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("📊 Live System Input")

cpu = st.sidebar.slider("CPU Usage", 0, 100, 50)
memory = st.sidebar.slider("Memory Usage", 0, 100, 50)
latency = st.sidebar.slider("Latency (ms)", 0, 300, 100)
packet = st.sidebar.slider("Packet Loss (%)", 0.0, 10.0, 1.0)
bandwidth = st.sidebar.slider("Bandwidth Usage", 100, 1000, 500)

# ---------------- KPIs ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Logs", len(df))
col2.metric("Active Incidents", int(df["Incident"].sum()))
col3.metric("System Health", "Stable" if df["Incident"].mean() < 0.5 else "Risky")

st.divider()

# ---------------- CHARTS ----------------
c1, c2 = st.columns(2)

with c1:
    fig1 = px.histogram(df, x="CPU_Usage", title="CPU Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    fig2 = px.histogram(df, x="Latency_ms", title="Latency Distribution")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------- PREDICTION ----------------
st.subheader("🔮 Incident Prediction Engine")

if st.button("Predict Incident"):

    input_data = np.array([[cpu, memory, latency, packet, bandwidth]])
    pred = model.predict(input_data)[0]

    if pred == 1:
        st.error("⚠ HIGH INCIDENT RISK DETECTED")
    else:
        st.success("✅ SYSTEM IS HEALTHY")

# ---------------- ROOT CAUSE ----------------
st.subheader("🧠 AI Root Cause Analysis")

if cpu > 85:
    st.warning("High CPU Usage detected → possible system overload")

elif memory > 90:
    st.warning("Memory exhaustion risk")

elif latency > 220:
    st.warning("Network congestion detected")

elif packet > 5:
    st.warning("Packet loss issue detected")

else:
    st.info("No critical issues detected")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💡 Built for AI Internship | Data Science + ML + AIOps Project")