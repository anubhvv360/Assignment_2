import streamlit as st

# Streamlit App Title
st.title("📊 SWOT Analysis - Quadrant View")

# Define SWOT categories
swot_data = {
    "Strengths": ["✅ Strong brand", "✅ High customer loyalty", "✅ Efficient supply chain"],
    "Weaknesses": ["❌ High production cost", "❌ Limited market reach", "❌ Dependence on suppliers"],
    "Opportunities": ["💡 Market expansion", "💡 Technological advancements", "💡 New partnerships"],
    "Threats": ["⚠️ Economic downturn", "⚠️ Competitive market", "⚠️ Regulatory changes"]
}

# Create a 2x2 layout using Streamlit columns
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Strengths (Top-Left)
with col1:
    st.markdown("### 💪 Strengths")
    st.info("\n".join(swot_data["Strengths"]))

# Weaknesses (Top-Right)
with col2:
    st.markdown("### 🚨 Weaknesses")
    st.warning("\n".join(swot_data["Weaknesses"]))

# Opportunities (Bottom-Left)
with col3:
    st.markdown("### 🌟 Opportunities")
    st.success("\n".join(swot_data["Opportunities"]))

# Threats (Bottom-Right)
with col4:
    st.markdown("### ⚡ Threats")
    st.error("\n".join(swot_data["Threats"]))

# Sidebar Insights
st.sidebar.header("ℹ️ Insights")
st.sidebar.info("📌 A SWOT quadrant helps visualize Strengths, Weaknesses, Opportunities, and Threats effectively.")
