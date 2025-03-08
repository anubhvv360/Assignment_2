import plotly.figure_factory as ff

# Define SWOT data
swot_table = [['Strengths', 'Weaknesses'],
              ['Strong brand', 'High production cost'],
              ['High customer loyalty', 'Limited market reach'],
              ['Efficient supply chain', 'Dependence on few suppliers']]

swot_table2 = [['Opportunities', 'Threats'],
              ['Market expansion', 'Economic downturn'],
              ['Technological advancements', 'Competitive market'],
              ['New partnerships', 'Regulatory changes']]

# Create a SWOT Table
fig1 = ff.create_table(swot_table, colorscale="greens")
fig2 = ff.create_table(swot_table2, colorscale="reds")

fig1.show()
fig2.show()
