import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Load model ─────────────────────────────
pkg = joblib.load("finance_model.pkl")
model = pkg['model']
features = pkg['features']
avg_burnout = pkg['avg_burnout']

# ── Page setup ─────────────────────────────
st.set_page_config(page_title="AI Finance Manager", layout="wide")

st.title("💰 AI Finance Manager")
st.subheader("Predict your financial burnout")

# ── Sidebar input ──────────────────────────
st.sidebar.header("Enter your monthly details")

income = st.sidebar.number_input("Income (₹)", value=60000)

food = st.sidebar.number_input("Food", value=7000)
rent = st.sidebar.number_input("Rent", value=15000)
travel = st.sidebar.number_input("Travel", value=3000)
entertainment = st.sidebar.number_input("Entertainment", value=4000)
health = st.sidebar.number_input("Health", value=1000)
shopping = st.sidebar.number_input("Shopping", value=5000)
utilities = st.sidebar.number_input("Utilities", value=2000)

# ── Calculations ───────────────────────────
expenses = {
    'food': food,
    'rent': rent,
    'travel': travel,
    'entertainment': entertainment,
    'health': health,
    'shopping': shopping,
    'utilities': utilities
}

total_exp = sum(expenses.values())

non_ess = entertainment + shopping + travel
ess = food + rent + utilities + health

burnout = total_exp / income if income > 0 else 0

# ── Create input for model ─────────────────
input_dict = {
    **expenses,
    'income': income,
    'total_expense': total_exp,
    'non_essential': non_ess,
    'essential': ess,
    'non_ess_ratio': non_ess / total_exp if total_exp else 0,
    'expense_per_income': burnout,
    'rent_ratio': rent / income if income else 0,
    'food_ratio': food / total_exp if total_exp else 0,
    'prev_burnout': avg_burnout,
    'rolling_3m_avg': total_exp
}
input_df = pd.DataFrame([input_dict])

# Add missing columns
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[features]

# ── Predict ───────────────────────────────
if st.button("🔍 Predict Burnout"):

    pred = model.predict(input_df)[0]
    pred = float(np.clip(pred, 0, 1))

    # ── Risk levels ───────────────────────
    if pred >= 0.90:
        level = "🚨 CRITICAL"
        advice = "Cut all non-essential spending immediately!"
        color = "red"
    elif pred >= 0.75:
        level = "⚠️ WARNING"
        advice = "Reduce shopping and entertainment."
        color = "orange"
    elif pred >= 0.55:
        level = "📊 MODERATE"
        advice = "Try increasing savings."
        color = "blue"
    else:
        level = "✅ HEALTHY"
        advice = "Excellent financial discipline!"
        color = "green"
st.subheader("📅 Monthly Spending Trend")

# Load your dataset (if saved)
try:
    df_hist = pd.read_csv("monthly_summary.csv")

    st.line_chart(df_hist[['total_expense', 'income']])

except:
    st.info("Monthly data not available")
    # ── Output ────────────────────────────
    st.subheader("📊 Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Burnout", f"{pred:.1%}")
        st.markdown(f"### {level}")
        st.write(f"💡 {advice}")

    with col2:
        st.metric("Total Expense", f"₹{total_exp:,}")
        st.metric("Savings", f"₹{income - total_exp:,}")

    # ── Charts ────────────────────────────
    st.subheader("📈 Expense Breakdown")

    df_exp = pd.DataFrame(expenses.items(), columns=['Category', 'Amount'])

    col3, col4 = st.columns(2)

    with col3:
        st.bar_chart(df_exp.set_index('Category'))

    with col4:
        st.write(df_exp)

    # ── Pie chart ─────────────────────────
    st.subheader("🥧 Spending Distribution")
    st.pyplot(df_exp.set_index('Category').plot.pie(
        y='Amount', autopct='%1.1f%%', figsize=(5,5)
    ).figure)
