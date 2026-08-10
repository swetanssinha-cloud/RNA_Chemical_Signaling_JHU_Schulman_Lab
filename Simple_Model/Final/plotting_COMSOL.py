import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data without headers and assign column names
df = pd.read_csv('COMSOL_S2_plot.txt', sep=r'\s+', header=None, names=['X', 'Y'])

# Plot the data
df.plot(x='X', y='Y', kind='line', marker='o')

# Add titles and show the graph
plt.title('Data From Text File')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.show()