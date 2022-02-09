##############################################################################
# For an overview of the code for this project, see the readme file.

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rc 
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator) 
from matplotlib.figure import figaspect 
import numpy.linalg as la 



### SETUP INFORMATION


# File names and variable separation parameter
fileStr = 'INSERT FILE NAME HERE.txt'
separationstr = '   ' 
saveStr_pc = 'prinicple_component_output.txt'
saveStr_centre = 'centred_principle_output.txt'
saveStr_deer_app = 'deep_app.txt'
 
# Font size parameters
SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
 
rc('font', **{'family': 'serif', 'serif': ['Arial']}) 
 
# Parameters for the plot 
n_trace = 6
pc_used = 3


### END OF SETUP INFORMATION

 
# Variable setup
data = np.loadtxt(fileStr, delimiter=separationstr, skiprows=1)
t = data[:, 0] 
deer = data[:, 1:] 
tmax = max(t)
 
# Plotting the traces
color = ['k', 'r', 'g', 'b', 'm', 'y'] 
startingSHC = np.eye(n_trace) 
plt.figure(figsize=(7,5))
for i in range(0, n_trace): 
    plt.plot(t, deer[:, i] + i)#, color[i])
    plt.title('Traces')
    plt.xlabel('t, ms')
    plt.ylabel('intensity, arb. units')
 
 
# %% PRINCIPAL COMPONENT ANALYSIS 

# SVD Calculation
u, s, vh = la.svd(deer[:, :]) 
m = u[:, :n_trace] * s 
 
 
w, h = figaspect(1.5) 
fig = plt.figure(figsize=(w, h)) 
 
# Plots prinicipal components and average values
for i in range(n_trace): 
    plt.plot(t[:], i * 1.1 + m[:, i], 'r') 
    plt.plot(t[:], i * 1.1 + 0 * t[:], 'k', linewidth=0.5) 
 
ax = plt.gca() 
ax.xaxis.set_major_locator(MultipleLocator(0.5)) 
ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
ax.tick_params('both', length=10, width=1, which='major') 
ax.tick_params('both', length=5, width=1, which='minor') 
plt.xlim([0, tmax]) 
plt.yticks([]) 
plt.title('Average Value and \n Principal Components')
plt.xlabel('t, ms')
plt.tight_layout() 
 

# Truncates splined trace dataset based on principal components
s_app = s 
smax = pc_used  #the number of principle component curves added to the curves 
s_app[smax:n_trace] = 0 
m_app = u[:, :n_trace] * s_app 
deer_app = np.dot(m_app, vh) 
shuffle = np.arange(n_trace) 
 
# Plots the principal components overlaid on top of the DEER traces
w, h = figaspect(1.5) 
fig = plt.figure(figsize=(w, h)) 
for i in range(n_trace): 
    plt.plot(t[:], shuffle[i] * 1.1 + deer_app[:, i], 'k') 
    plt.plot(t[:], shuffle[i] * 1.1 + deer[:, i], 'r') 

# Axis setup and labelling
ax = plt.gca() 
ax.xaxis.set_major_locator(MultipleLocator(0.5)) 
ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
ax.tick_params('both', length=10, width=1, which='major') 
ax.tick_params('both', length=5, width=1, which='minor') 
plt.xlim([0, tmax])
plt.ylim([-1,8]) 
plt.yticks([]) 
if pc_used == 1:
    plt.title('Traces Overlaid with ' + np.str(smax) + '\n Principal Component')
else:
    plt.title('Traces Overlaid with ' + np.str(smax) + '\n Principal Components')
plt.xlabel('t, ms') 
plt.tight_layout()

# Saves the m matrix to a file along with the time values
pltmatrix = np.zeros((len(t), n_trace+1))
for i in range(0, len(t)):
    for n in range(n_trace+1):
        if n == 0:
            pltmatrix[i, n] = t[i]
        else:
            pltmatrix[i, n] = m[i, n-1]
np.savetxt(saveStr_pc, pltmatrix, delimiter=separationstr)

# Saves the deer matrix to a file along with the time values
pltmatrix_deer = np.zeros((len(t), n_trace+1))
for i in range(0, len(t)):
    for n in range(n_trace+1):
        if n == 0:
            pltmatrix_deer[i, n] = t[i]
        else:
            pltmatrix_deer[i, n] = deer_app[i, n-1]
np.savetxt(saveStr_deer_app, pltmatrix_deer, delimiter=separationstr)
 
# %% CENTERED PCA

# Centres the data and then performs SVD
xmean = np.mean(deer, axis=1) 
X = deer.transpose() - xmean.transpose() 
X = X.transpose() 
u, s, vh = la.svd(X) 
m_c = u[:, :n_trace] * s 

# Plots the centred principal curves 
w, h = figaspect(1.5) 
fig = plt.figure(figsize=(w, h)) 
for i in range(n_trace): 
    plt.plot(t[:], i * 1.1 + m_c[:, i], 'r') 
    plt.plot(t[:], i * 1.1 + 0 * t[:], 'k', linewidth=0.5) 
 
# Axis setup and labelling
plt.plot(t[:], -1.1 + xmean, 'r') 
plt.plot(t[:], -1.1 + 0* t[:], 'k', linewidth=0.5) 
plt.title("Average Value and \n Principal Components \n (Centred)") 
plt.xlabel("t, ms")
ax = plt.gca() 
ax.xaxis.set_major_locator(MultipleLocator(0.5)) 
ax.xaxis.set_minor_locator(MultipleLocator(0.1)) 
ax.tick_params('both', length=10, width=1, which='major') 
ax.tick_params('both', length=5, width=1, which='minor') 
plt.xlim([0, tmax]) 
plt.yticks([]) 
plt.tight_layout() 

# Saves the centred m matrix to a file alogn with the time values
pltmatrix_c = np.zeros((len(t), n_trace+1))
for i in range(0, len(t)):
    for n in range(n_trace+1):
        if n == 0:
            pltmatrix_c[i, n] = t[i]
        else:
            pltmatrix_c[i, n] = m_c[i, n-1]
np.savetxt(saveStr_centre, pltmatrix_c, delimiter=separationstr)