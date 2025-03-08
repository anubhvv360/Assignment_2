import streamlit as st
import plotly.graph_objects as go

# Define SWOT categories
swot_data = {
    "Strengths": ["Strong brand", "High customer loyalty", "Efficient supply chain"],
    "Weaknesses": ["High production cost", "Limited market reach", "Dependence on few suppliers"],
    "Opportunities": ["Market expansion", "Technological advancements", "New partnerships"],
    "Threats": ["Economic downturn", "Competitive market", "Regulatory changes"]
}

# Streamlit App Title
st.title("📊 SWOT Analysis with Plotly")

# Create a SWOT Table
header = ["Category", "Details"]
values = [
    ["✅ Strengths", "<br>".join(swot_data["Strengths"])],
    ["❌ Weaknesses", "<br>".join(swot_data["Weaknesses"])],
    ["💡 Opportunities", "<br>".join(swot_data["Opportunities"])],
    ["⚠️ Threats", "<br>".join(swot_data["Threats"])]
]

fig = go.Figure(data=[go.Table(
    header=dict(values=header, fill_color='royalblue', font=dict(color='white', size=14), align="left"),
    cells=dict(values=[list(zip(*values))[0], list(zip(*values))[1]],
               fill_color=[["lightgreen", "lightcoral", "lightblue", "lightyellow"]],
               align="left", font=dict(size=12))
)])

# Display the SWOT Table
st.plotly_chart(fig)

# Add interactive insights
st.sidebar.header("ℹ️ Insights")
st.sidebar.info("💡 A well-defined SWOT analysis helps in strategic planning.")

