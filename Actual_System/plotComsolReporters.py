import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

numRows = 3  # 3 data columns

plt.rcParams["figure.figsize"] = (5,3)

def getCurves(fileName):
    t = []
    p = []
    for ii in range(numRows):
        p.append([])
    kk = 0
    header = 5
    f=open(fileName) 

    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        kk += 1
        jj = 0
        if kk > header:
            for ii in row:
                if str(ii) != "":
                    jj += 1
                    if jj==1:
                        t.append(float(ii))
                    elif jj>1 and jj<numRows+2:
                        p[jj-2].append(float(ii))

    t = np.array(t)/60
    p = np.array(p)
    
    # Convert from mol/m³ to nM
    p = p * 1e6  # Multiply by 1,000,000
    
    # Plot 1: rTh (threshold) only
    plt.figure(figsize=(5,3))
    plt.plot(t, p[1], '-', color='blue', label="rTh (Bound threshold)")
    plt.xlabel("Time (min)")
    plt.ylabel("Concentration (nM)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot 2: rS6 (sender) and rGRep6 (reporter)
    plt.figure(figsize=(5,3))
    cmap = matplotlib.colormaps['plasma']
    plt.plot(t, p[0], '-')
    plt.xlabel("Time (min)")
    plt.ylabel("Concentration of free sender (nM)")
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Sum of all three
    plt.figure(figsize=(5,3))
    total = p[0] + p[1] + p[2]
    plt.plot(t, total, '-', color='green', label="Total (rS6 + rTh + rGRep6)")
    plt.xlabel("Time (min)")
    plt.ylabel("Total S2Concentration (nM)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot 4: [I2] = Initial reporter - Current reporter
    plt.figure(figsize=(5,3))
    initial_reporter = p[2][0]  # Initial value of rGRep6
    I2 = initial_reporter - p[2]  # Initial minus current
    plt.plot(t, I2, '-', color='red', label="[I2]")
    plt.xlabel("Time (hrs)")
    plt.ylabel("Concentration of I2 (nM)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return t, p

getCurves("Single_sender_receiver-200_um.txt")