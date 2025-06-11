import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

beam_sizes = [3, 5, 10, 15]
durations = [23.41, 27.91, 26.92, 28.42]      # Duration in hours
emissions = [188.71, 341.27, 563.12, 765.63]  # Emissions in CO₂ eq

x = np.arange(len(beam_sizes))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 5))

bars1 = ax1.bar(x - width/2, durations, width, label='Duration (hrs)', color='tab:blue')
ax1.set_ylabel('Duration (hours)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, emissions, width, label='Emissions (CO₂ eq)', color='tab:green')
ax2.set_ylabel('Emissions (CO₂ eq)', color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}',
             ha='center', va='bottom', fontsize=9, color='tab:blue')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 10, f'{height:.1f}',
             ha='center', va='bottom', fontsize=9, color='tab:green')

plt.xticks(x, [f'k={k}' for k in beam_sizes])
ax1.set_xlabel('Beam size ($k$)')

fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

plt.tight_layout()
plt.show()
