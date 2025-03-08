import plotly.figure_factory as ff

# Define SWOT data
swot_data = [
    ["🟢 Strengths", "🔴 Weaknesses"],
    ["Strong brand recognition", "High production costs"],
    ["Loyal customer base", "Limited market reach"],
    ["Efficient supply chain", "Dependence on few suppliers"]
]

swot_data2 = [
    ["🔵 Opportunities", "⚠️ Threats"],
    ["Market expansion", "Economic downturn"],
    ["Technological advancements", "Competitive market"],
    ["New strategic partnerships", "Regulatory changes"]
]

# Create tables for SWOT analysis
fig1 = ff.create_table(swot_data, colorscale="greens")
fig2 = ff.create_table(swot_data2, colorscale="reds")

# Show the first table (Strengths & Weaknesses)
fig1.show()

# Show the second table (Opportunities & Threats)
fig2.show()
