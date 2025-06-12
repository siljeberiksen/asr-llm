import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Graph showing growth in parameters for LLM

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


df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'], format="%d %B %Y", errors="coerce")
df = df.sort_values('Date')

fig, ax = plt.subplots(figsize=(12, 6))

formatter = mdates.DateFormatter("%Y") 
locator = mdates.YearLocator() 
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(locator)
plt.yscale('log')
ax.plot(df['Date'], df['Parameters (B)'], marker='o', label="Selected LLMs")

for i, (model, params, date) in enumerate(zip(df['Model'], df['Parameters (B)'], df["Date"])):
    offset = 30 if i % 2 == 0 else 15 
    if model == "Megatron-LM":
        offset = 65
    if model == "T5":
        offset = 80
    if model == "PaLM":
        offset = -15

    if model == "LLaMA":
        offset = -20

    if model == "LLaMA 3":
        offset = -20

    if model =="Chinchilla":
        offset = -20

    ax.annotate(
        f"{model}",
        (date, params),
        textcoords="offset points",
        xytext=(0, offset),
        ha="center",
        fontsize=15,
        arrowprops=dict(
            arrowstyle="-", 
            color="gray",
            lw=1,
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
degree = 2
x = df['Date'].map(lambda d: d.toordinal()).values
y = df['Parameters (B)'].values

coefficients = np.polyfit(x,y , degree)

x_fit = np.linspace(min(x), max(x), 500)  # Generate 500 points for a smooth curve
y_fit = np.polyval(coefficients, x_fit)
y_fit = np.maximum(y_fit, 0)

x_fit_dates = [pd.Timestamp.fromordinal(int(date)) for date in x_fit]
ax.plot(x_fit_dates, y_fit, color='red', label=f'Polynomial fit (degree {degree})', zorder=2)

ax.set_xlabel("Year", fontdict={"fontsize":15})
ax.set_ylabel("Parameters (in billions)", fontdict={"fontsize":15})

ax.tick_params(axis='x', labelsize=15) 
ax.tick_params(axis='y', labelsize=15) 
ax.legend(loc='upper left', fontsize=15) 

ax.grid(True)
plt.tight_layout()



# Word embedding example

plt.figure(figsize=(6, 6))
x, y = 1, 9
plt.annotate(f"Grandfather", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black')
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1) 
plt.scatter([x], [y], color='red', s=100, label="Endpoint")

x, y = 1, 7
plt.annotate(f"Man", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black')
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1) 
plt.scatter([x], [y], color='red', s=100, label="Endpoint")

x, y = 5, 7
plt.annotate(f"Adult", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black') 
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1) 
plt.scatter([x], [y], color='red', s=100, label="Endpoint")


x, y = 9, 7
plt.annotate(f"Woman", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black')
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1) 
plt.scatter([x], [y], color='red', s=100, label="Endpoint")

x, y = 1, 2
plt.annotate(f"Boy", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black')
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1) 
plt.scatter([x], [y], color='red', s=100, label="Endpoint")

x, y = 5, 2
plt.annotate(f"Child", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black')
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1, linewidth=0.5) 
plt.scatter([x], [y], color='red', s=100, label="Endpoint")

x, y = 9, 2
plt.annotate(f"Girl", (x, y), textcoords="offset points", xytext=(10, 10), ha='center', color='black')
plt.quiver(0, 0, x, y, color='b', units='xy', scale=1) 
plt.scatter([x], [y], color='red', s=100, label="Endpoint")
  
plt.xlim(0, 10) 
plt.ylim(0, 10) 

plt.xlabel("Gender")
plt.ylabel("Age")
  
plt.grid() 
plt.show() 
