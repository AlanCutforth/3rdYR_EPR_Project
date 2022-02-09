##############################################################################
# For an overview of the code for this project, see the readme file.

from scipy.interpolate import CubicSpline
import numpy as np
import collections
import matplotlib.pyplot as plt

# Number of graphs
g_count = 6
# Optional Curve labels - leave blank for automatic labelling
labels = ['Curve 1', 'Curve 2']


# String to separate numerical values
separationstr_wr = "   "
separationstr_r = "   "

# Creating the column labels for the output file
if len(labels) != g_count:
    autolabels = []
    for i in range(0, g_count):
        autolabels.append("A_" + str(i+1))
else:
    autolabels = labels

# Puts the data into a matrix, ensuring the matrix can be created by filling blank spaces with NaN values
def initialise_data(t, y, n):
    global t_m
    global a_m
    
    for i in range(0, len(t)):
        t_m[n, i] = t[i]
        a_m[n, i] = y[i]
    
    if len(t) < pt_max:
        for i in range(len(t), pt_max):
            t_m[n, i] = np.NaN
            a_m[n, i] = np.NaN
            
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
        
# Builds the row strings for printing into the file            
def build_wr_str(t, a, n):
    global g_count
    global separationstr_wr
    
    wr_str = str(t[n]) + separationstr_wr
    
    for i in range(0, g_count):
        wr_str = wr_str + str(a[i])
        if i != g_count-1:
            wr_str = wr_str + separationstr_wr
        else:
            wr_str = wr_str + "\n"
    
    return wr_str

# Splits a string at a specific index
def split_string(str_i, sep, n):
    sep_ind = str_i.find(sep)
    if n == 1:
        return str_i[0:sep_ind]
    else:
        return str_i[(sep_ind + len(sep)):(len(str_i) - 1)]

tmin = 0
tmax = 0
pt_max = 0

# Find the maximum value of points
for i in range(0, g_count):
    fileStr = '1.DEER_center0_' + str(i+1) + '.txt'
    count = file_len(fileStr)
        
    if count > pt_max:
        pt_max = count

# Set up the data matrices
t_m = np.zeros((g_count, pt_max))
a_m = np.zeros((g_count, pt_max))

# Reads the data into arrays, and by calling the initialise function adds the data to the matrices
for i in range(0, g_count):
    readData_t = []
    readData_a = []
    fileStr = '1.DEER_center0_' + str(i+1) + '.txt'
    fileRead = open(fileStr, 'r').readlines()
    for n in range(0, len(fileRead)):
        readData_t.append(float(split_string(fileRead[n], separationstr_r, 1)))
        readData_a.append(float(split_string(fileRead[n], separationstr_r, 2)))
    
    if max(readData_t) > tmax:
        tmax = max(readData_t)
    
    initialise_data(readData_t, readData_a, i)
    

# Plots the data
ts = np.arange(tmin, tmax, 0.001)
plt.figure()
for i in range(g_count):
    plt.plot(t_m[i,:], a_m[i,:] + i)

splines = []

# Performs the cubic spline for all of the data
for n in range(0, g_count):
    current_set_t = t_m[n,:].tolist()
    current_set_a = a_m[n,:].tolist()

    for i in range(pt_max-1, 0, -1):
        if np.isnan(current_set_t[i]):
            del current_set_t[i]
            del current_set_a[i]

    current_set_t.sort()
    
    #Finds the index at which a duplicate is initially found
    repeatx = [item for item, count in collections.Counter(current_set_t).items() if count > 1] 
    while len(repeatx) > 0:
        ind = current_set_t.index(repeatx[0])
        
        #Loop repeats until there are no more duplicates
        current_set_a.remove(current_set_a[ind])
        current_set_t.remove(current_set_t[ind])
        repeatx = [item for item, count in collections.Counter(current_set_t).items() if count > 1]
    
    #Performing cubic spline
    splines.append(CubicSpline(current_set_t, current_set_a))


# Filewriting Part

fy = open("deer_output.txt", "a")
sp_af = np.zeros((g_count, int(tmax*1000)+1))

namestr = "TIME:" + separationstr_wr

for n in range(0, g_count):
    current_af = splines[n](ts)
    for i in range(0, len(ts)):
        sp_af[n,i] = current_af[i]
        
    if n == g_count -1:
        namestr = namestr + autolabels[n] + "\n"
    else:
        namestr = namestr + autolabels[n] + separationstr_wr
        
fy.write(namestr)

for i in range(0, len(ts)):
    print_vector = []
    for n in range(0, g_count):
        print_vector.append(sp_af[n,i])
        
    fy.write(build_wr_str(ts, print_vector, i))
fy.close()

plt.figure()
for i in range(g_count):
    plt.plot(ts, sp_af[i,:] + i)

