import streamlit as st
import plotly.graph_objects as go

# Streamlit App Title
st.title("📊 SWOT Analysis - Quadrant View")

# Define SWOT categories
swot_data = {
    "Strengths": ["Strong brand", "High customer loyalty", "Efficient supply chain"],
    "Weaknesses": ["High production cost", "Limited market reach", "Dependence on suppliers"],
    "Opportunities": ["Market expansion", "Technological advancements", "New partnerships"],
    "Threats": ["Economic downturn", "Competitive market", "Regulatory changes"]
}

# Define quadrant positions for SWOT
swot_positions = {
    "Strengths": (0.25, 0.75),
    "Weaknesses": (0.75, 0.75),
    "Opportunities": (0.25, 0.25),
    "Threats": (0.75, 0.25),
}

# Create figure
fig = go.Figure()

# Add the four quadrants
fig.add_shape(type="rect", x0=0, y0=0.5, x1=0.5, y1=1, fillcolor="lightgreen", opacity=0.3, line=dict(width=0))  # Strengths
fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=1, y1=1, fillcolor="lightcoral", opacity=0.3, line=dict(width=0))  # Weaknesses
fig.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=0.5, fillcolor="lightblue", opacity=0.3, line=dict(width=0))  # Opportunities
fig.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=0.5, fillcolor="lightyellow", opacity=0.3, line=dict(width=0))  # Threats

# Add SWOT labels
for category, (x, y) in swot_positions.items():
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        text=[f"<b>{category}</b><br>" + "<br>".join(swot_data[category])],
        mode="text",
        textposition="middle center",
        hoverinfo="text"
    ))

# Update layout
fig.update_layout(
    xaxis=dict(showticklabels=False, zeroline=False, range=[0, 1]),
    yaxis=dict(showticklabels=False, zeroline=False, range=[0, 1]),
    showlegend=False,
    title="SWOT Analysis Quadrant"
)

# Display in Streamlit
st.plotly_chart(fig)

# Sidebar Insights
st.sidebar.header("ℹ️ Insights")
st.sidebar.info("📌 A SWOT quadrant helps visualize Strengths, Weaknesses, Opportunities, and Threats effectively.")
