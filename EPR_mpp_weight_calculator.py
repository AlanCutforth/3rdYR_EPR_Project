##############################################################################
# For an overview of the code for this project, see the readme file.

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rc 
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator) 
from matplotlib.figure import figaspect 
import numpy.linalg as la 
import orisel as os
from sklearn import linear_model


def delete_rows(mat, spacing):
    n_r = len(mat)/spacing
    n_c = len(mat[0])
    new_mat = np.zeros((int(len(mat)-n_r),int(n_c)))

    j = 0
    for i in range(len(mat)):
        if i % spacing != 0 or i == 0:
            for n in range(n_c):
                print(str(i) + ", " + str(j))
                new_mat[j-1, n] = mat[i, n]
            j += 1
        
    return new_mat
            
    
# File names and variable separation parameter
fileStr = 'DEER_center0.txt'
separationstr = '   ' 
saveStr_pc = 'prinicple_component_output.txt'
saveStr_centre = 'centred_principle_output.txt'
saveStr_deer_app = 'deep_app.txt'
 
 
SMALL_SIZE = 24 
MEDIUM_SIZE = 10 
BIGGER_SIZE = 12 
 
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes 
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title 
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels 
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels 
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels 
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize 
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title 
 
rc('font', **{'family': 'serif', 'serif': ['Arial']}) 
 
# Number of traces
n_trace = 6
# Number of meaningful principle components
n_bas = 3
# Delete every x point
delete_spacing = 2
delete_any_points = True


data_load = np.loadtxt(fileStr, delimiter=separationstr, skiprows=1)

if delete_any_points == True:
    data = delete_rows(data_load, delete_spacing)
    data = delete_rows(data_load, delete_spacing)
    data = delete_rows(data_load, delete_spacing)
else:
    data = data_load


t = data[:, 0] 
i_deer = data[:, 1:] 
tmax = max(t) 


n_trace = len(data[0]) - 1

r = np.linspace(2,6,150)
f = os.gaussian(r,3.75,0.13)
 
 
startingSHC = np.eye(n_trace) 
for i in range(0, n_trace): 
    plt.plot(t, i_deer[:, i] + i)
    plt.title("Traces")


deer=np.zeros_like(i_deer)
fit = np.exp(-t*1e-3*0.02)
for i in range(n_trace):
    #deer[:,i] =  (i_deer[:,i]/fit - 0.98)/0.02
    deer[:,i] = i_deer[:,i]

#%% finding optimal MPP weights
n_time = len(t)
n_r = len(r)

basis_kernel = np.zeros((n_bas,n_time, n_r))
i = 0
for v in np.eye(n_bas,n_bas):
    K = os.make_kernel(t, r, v)
    basis_kernel[i,:,:] = K
    i = i + 1

p = startingSHC
kernel = np.zeros( (n_trace, n_time, n_r))
full_kernel = np.reshape(np.ravel(kernel),(n_trace*n_time,n_r))
full_trace = np.transpose(deer)
full_trace = full_trace.ravel()

x = f
alpha = 1
#startingSHC
starting_weights = np.eye(n_trace, n_bas)

reg = linear_model.Ridge(alpha = 1, fit_intercept = False)
#linfit = linear_model.Ridge(alpha = 1e-6)
trace = np.zeros_like(deer)

rnorm_full = []
for k in range(1000):

    A = basis_kernel@x
    new_weights = np.zeros_like(starting_weights)    

    
    for i in range(n_trace):
        b = deer[:,i]
        reg.fit(A.transpose(), b)
        new_weights[i,:] = reg.coef_

    p = new_weights    
    
    for j in range(n_trace):
        N =np.zeros((n_time,n_r))
        for i in range(n_bas):
            N = N + p[j,i]*basis_kernel[i,:,:]

        kernel[j,:,:] = N        
    trace = deer
    
    full_kernel = np.reshape(np.ravel(kernel),(n_trace*n_time,n_r))
    full_trace = np.transpose(trace)
    full_trace = full_trace.ravel()
    x, rnorm = os.solve_tikhonov(full_kernel, full_trace, alpha)
    print (rnorm)
    rnorm_full.append(rnorm)



plt.figure()
plt.plot(full_kernel@x)
plt.plot(full_trace)
plt.figure()
plt.plot(r,x)
plt.figure()
plt.plot(np.asarray(rnorm_full))
x_max_index = 0
x_max = max(x)
for i in range(len(x)):
    if x[i] == x_max:
        x_max_index = i
print('Distance distribution peak at ' + str(r[x_max_index]) + 'nm.')


w, h = figaspect(2)
fig = plt.figure(figsize=(w, h))
shuffle = [0, 1, 2, 4, 3, 5]
full2 = np.reshape(full_kernel@x, (n_trace, n_time))
full2 = np.transpose(full2)
shift = np.linspace(0, 6, 6)
for i in range(6):
    plt.plot(t/1, shift[shuffle[i]] + trace[:, i])
    plt.plot(t/1, shift[shuffle[i]] + full2[:, i])
plt.yticks([])
plt.xlim([0, 2.36])
ax = plt.gca()
ax.tick_params('both', length=10, width=1, which='major')
ax.tick_params('both', length=5, width=1, which='minor')
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
plt.xticks(np.arange(0, 2.36, 0.5))
plt.savefig('iterative_fitting_deer.svg', transparent = True)
plt.savefig('FigS3c.png', transparent = True)
plt.savefig('FigS3c.svg', transparent = True)

plt.figure()
plt.plot((np.transpose(new_weights)/np.transpose(new_weights)[0,:]),'s-')

