import pandas as pd
import plotly.graph_objects as go

# Read the CSV file into a DataFrame
df = pd.read_csv("Experiments40/Pidgeon.csv")

# Define colors for each algorithm
colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]

# Create an empty figure for Time vs Budget
fig_time_vs_budget = go.Figure()

# Loop through each algorithm
for i, algorithm in enumerate(df["Algorithm"].unique()):
    # Select data for the current algorithm
    algorithm_data = df[df["Algorithm"] == algorithm]
    # Loop through each curve within the algorithm
    for j, curve in enumerate(algorithm_data["Curves"].unique()):
        # Select data for the current curve
        curve_data = algorithm_data[algorithm_data["Curves"] == curve]
        # Sort the data by Budget
        curve_data = curve_data.sort_values("Budget")
        # Assign color to the curve
        color = colors[i % len(colors)]
        # Add scatter plot for Budget vs Time
        fig_time_vs_budget.add_trace(
            go.Scatter(
                x=curve_data["Budget"],
                y=curve_data["Time"],
                mode="markers+lines",
                marker=dict(color=color),
                name=f"{algorithm} - {curve}",
            )
        )

# Update layout for Time vs Budget
fig_time_vs_budget.update_layout(
    title="Time vs Budget",
    xaxis=dict(title="Budget", type="log"),
    yaxis=dict(title="Time", type="log"),
)

# Save plot for Time vs Budget as HTML
fig_time_vs_budget.write_html("time_vs_budget.html")

# Show plot for Time vs Budget
fig_time_vs_budget.show()

# Create an empty figure for Radius vs Budget
fig_radius_vs_budget = go.Figure()

# Loop through each algorithm
for i, algorithm in enumerate(df["Algorithm"].unique()):
    # Select data for the current algorithm
    algorithm_data = df[df["Algorithm"] == algorithm]
    # Loop through each curve within the algorithm
    for j, curve in enumerate(algorithm_data["Curves"].unique()):
        # Select data for the current curve
        curve_data = algorithm_data[algorithm_data["Curves"] == curve]
        # Sort the data by Budget
        curve_data = curve_data.sort_values("Budget")
        # Assign color to the curve
        color = colors[i % len(colors)]
        # Add scatter plot for Budget vs Radius
        fig_radius_vs_budget.add_trace(
            go.Scatter(
                x=curve_data["Budget"],
                y=curve_data["Radius"],
                mode="markers+lines",
                marker=dict(color=color),
                name=f"{algorithm} - {curve}",
            )
        )

# Update layout for Radius vs Budget
fig_radius_vs_budget.update_layout(
    title="Radius vs Budget",
    xaxis=dict(title="Budget", type="log"),
    yaxis=dict(title="Radius", type="log"),
)

# Save plot for Radius vs Budget as HTML
fig_radius_vs_budget.write_html("radius_vs_budget.html")

# Show plot for Radius vs Budget
fig_radius_vs_budget.show()
