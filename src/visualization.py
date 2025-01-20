# import pandas as pd

# # Data
# data = [
#     {"Date": "May 2018", "Model": "ELMo", "Parameters": "93M"},
#     {"Date": "11.06.2018", "Model": "GPT", "Parameters": "117M (secondary source)"},
#     {"Date": "11 October 2018", "Model": "BERT (Large)", "Parameters": "340M"},
#     {"Date": "14 February 2019", "Model": "GPT-2", "Parameters": "1.5B"},
#     {"Date": "17 September 2019", "Model": "Megatron-LM", "Parameters": "8.3B"},
#     {"Date": "23 October 2019", "Model": "T5", "Parameters": "11B"},
#     {"Date": "13 February 2020", "Model": "Turing NLG", "Parameters": "14.2B"},
#     {"Date": "28 May 2020", "Model": "GPT-3", "Parameters": "175B"},
#     {"Date": "4 April 2022", "Model": "PaLM", "Parameters": "540B"},
#     {"Date": "20 January 2022", "Model": "LaMDA", "Parameters": "137B"},
#     {"Date": "27 February 2023", "Model": "LLaMA", "Parameters": "65.2B"},
#     {"Date": "18 July 2023", "Model": "LLaMA 2", "Parameters": "70B"},
#     {"Date": "31 July 2024", "Model": "LLaMA 3", "Parameters": "405B"},
# ]

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Save as CSV
# df.to_csv("llm_history.csv", index=False)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Data
#{"Date": "May 2018", "Model": "ELMo", "Parameters (B)": 0.093},
data = [
    #{"Date": "15 February 2018", "Model": "ELMo", "Parameters (B)": 0.093},
    {"Date": "11 June 2018", "Model": "GPT", "Parameters (B)": 0.117},
    {"Date": "11 October 2018", "Model": "BERT", "Parameters (B)": 0.340},
    {"Date": "14 February 2019", "Model": "GPT-2", "Parameters (B)": 1.5},
    {"Date": "17 September 2019", "Model": "Megatron-LM", "Parameters (B)": 8.3},
    {"Date": "23 October 2019", "Model": "T5", "Parameters (B)": 11.0},
    {"Date": "13 February 2020", "Model": "Turing NLG", "Parameters (B)": 14.2},
    {"Date": "28 May 2020", "Model": "GPT-3", "Parameters (B)": 175.0},
    {"Date": "4 April 2022", "Model": "PaLM", "Parameters (B)": 540.0},
    {"Date": "20 January 2022", "Model": "LaMDA", "Parameters (B)": 137.0},
    {"Date": "29 March 2022", "Model": "Chinchilla", "Parameters (B)": 70.0},
    {"Date": "27 February 2023", "Model": "LLaMA", "Parameters (B)": 65.2},
    {"Date": "18 July 2023", "Model": "LLaMA 2", "Parameters (B)": 70.0},
    {"Date": "13 March 2024", "Model": "Gemma", "Parameters (B)": 7.0},
    {"Date": "31 July 2024", "Model": "LLaMA 3", "Parameters (B)": 405.0},
    {"Date": "2 october 2024", "Model": "Gemma 2", "Parameters (B)": 27.0}
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert "Date" to datetime for proper sorting and handling
df['Date'] = pd.to_datetime(df['Date'], format="%d %B %Y", errors="coerce")

# Ensure the data is sorted by date
df = df.sort_values('Date')

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

# Format the x-axis for years only
formatter = mdates.DateFormatter("%Y")  # Format x-axis as years
locator = mdates.YearLocator()  # Show ticks at yearly intervals
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(locator)
plt.yscale('log')
# Plot the data
ax.plot(df['Date'], df['Parameters (B)'], marker='o', label="Parameters Growth")

# Annotate each point with a label
for i, (model, params, date) in enumerate(zip(df['Model'], df['Parameters (B)'], df["Date"])):
    # Adjust offsets for overlapping labels
    offset = 30 if i % 2 == 0 else 15  # Alternate offsets
    if model == "Megatron-LM":
        offset = 65
    if model == "T5":
        offset = 80
    if model == "PaLM":
        offset = -15

    if model == "LLaMA 3":
        offset = -20

    if model =="Chinchilla":
        offset = -20

    ax.annotate(
        f"{model}",
        (date, params),
        textcoords="offset points",
        xytext=(0, offset),  # Adjusted offset
        ha="center",
        fontsize=15,
        arrowprops=dict(
            arrowstyle="-",  # Simple line
            color="gray",
            lw=1,  # Line width
            alpha=1.0,
            zorder=1 
        ),
        bbox=dict(
            boxstyle="round",
            edgecolor="black",
            facecolor="white",
            alpha=1,  
            zorder=2
        )
    )
# Fit a polynomial of degree 2 (quadratic) to the data
degree = 2

# Convert dates to numeric values (e.g., ordinal format for fitting)
x = df['Date'].map(lambda d: d.toordinal()).values
y = df['Parameters (B)'].values

coefficients = np.polyfit(x,y , degree)


# Generate a smooth curve for the polynomial
x_fit = np.linspace(min(x), max(x), 500)  # Generate 500 points for a smooth curve
y_fit = np.polyval(coefficients, x_fit)
# Ensure all y_fit values are greater than 0
y_fit = np.maximum(y_fit, 0)

# Convert x_fit back to datetime for plotting
x_fit_dates = [pd.Timestamp.fromordinal(int(date)) for date in x_fit]
ax.plot(x_fit_dates, y_fit, color='red', label=f'Polynomial Fit (Degree {degree})', zorder=2)

# Labels, title, and grid
ax.set_xlabel("Year", fontdict={"fontsize":15})
ax.set_ylabel("Parameters (in Billions)", fontdict={"fontsize":15})
#ax.set_title("Growth of Language Model Parameters (2018-2024)", fontsize = 20)

ax.tick_params(axis='x', labelsize=15)  # Font size for x-axis values
ax.tick_params(axis='y', labelsize=15)  # Font size for y-axis values
ax.legend(loc='upper left', fontsize=15) 

ax.grid(True)

# Tight layout for better spacing
plt.tight_layout()

# Show plot
#plt.show()


# Import libraries 
import numpy as np 
import matplotlib.pyplot as plt 
  
# Vector origin location 
  
# # Directional vectors 
# U = [1]   
# V = [9]  


plt.figure(figsize=(6, 6))
# Annotate the endpoint
x, y = 1, 9
plt.annotate(f"Grandfather", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black')
# Creating plot 
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1) 
# Plot a round point at the endpoint
plt.scatter([x], [y], color='red', s=100, label="Endpoint")

x, y = 1, 7
plt.annotate(f"Man", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black')
# Creating plot 
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1) 
# Plot a round point at the endpoint
plt.scatter([x], [y], color='red', s=100, label="Endpoint")

x, y = 5, 7
plt.annotate(f"Adult", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black')
# Creating plot 
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1) 
# Plot a round point at the endpoint
plt.scatter([x], [y], color='red', s=100, label="Endpoint")


x, y = 9, 7
plt.annotate(f"Woman", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black')
# Creating plot 
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1) 
# Plot a round point at the endpoint
plt.scatter([x], [y], color='red', s=100, label="Endpoint")

x, y = 1, 2
plt.annotate(f"Boy", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black')
# Creating plot 
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1) 
# Plot a round point at the endpoint
plt.scatter([x], [y], color='red', s=100, label="Endpoint")

x, y = 5, 2
plt.annotate(f"Child", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black')
# Creating plot 
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1, linewidth=0.5) 
# Plot a round point at the endpoint
plt.scatter([x], [y], color='red', s=100, label="Endpoint")

x, y = 9, 2
plt.annotate(f"Girl", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black')
# Creating plot 
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1) 
# Plot a round point at the endpoint
plt.scatter([x], [y], color='red', s=100, label="Endpoint")
  
  
# x-lim and y-lim 
plt.xlim(0, 10) 
plt.ylim(0, 10) 



# Labels and legend
plt.xlabel("Gender")
plt.ylabel("Age")
  
# Show plot with grid 
plt.grid() 
plt.show() 
