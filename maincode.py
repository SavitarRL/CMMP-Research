# code for num eq
import random
import math as m
import numpy as np
from numpy.linalg import norm
import scipy
import sympy
import scipy.integrate as spint
from sympy import diff, symbols
import matplotlib.pyplot as plt
from scipy import special
import statistics
import csv

## main equations pbtained from derivations and Mean-Field Approximation
def A(b,l,a):
    return b*(1+2*l*(1+a))

def B(t,a):
    return 1-2*t*(1+a)

def E(l , t , a ,b):
    integral = -1/(np.pi) * np.abs(A(b,l,a)+B(t,a)) * special.ellipe((4*A(b,l,a)*B(t,a))/(A(b,l,a)+B(t,a))**2)
    term = (1+a)*(b*l**2 - t**2)
    return integral + term

def delE_l(l , t , a ,b, steps=0.00001):
    return (E(l+steps , t , a ,b)-E(l , t , a ,b))/steps

def delE_t(l , t , a ,b, steps=0.00001):
    return (E(l , t+steps , a ,b)-E(l , t , a ,b))/steps

def gradE(l, t, a, b):
        return np.array([delE_l(l, t, a, b), delE_t(l, t, a, b)])

    
## finding suitable expectation values lambda and t

## gradient descent for lambda and ascend for t 
def descent(l,t,a,b,learn_rate = 0.1,mxloop=1000, tolerance=5):
    n=0
    while n < mxloop:
#         print([l,t], E(l, t, a,b), gradE(l, t, a,b))
        if round(gradE(l, t, a,b)[0],tolerance) != 0 and round(gradE(l, t, a,b)[1], tolerance) != 0:
            l -= learn_rate * gradE(l, t, a,b)[0]
            t += learn_rate * gradE(l, t, a,b)[1]
            n+=1   
        else:
            return [l,t]
    return [l,t] ## assuming it reaches golbal/local minimum after that many steps
### finding values of l and t
def E_root(a,b): 
    l = random.uniform(0, 1)
    t = random.uniform(0, 1)
    l_list = []
    t_list = []
    return descent(l,t,a,b)

### Ploting the Energy density of given alpha and beta over a range of expectation values lambda and t
def PlotEnergy(a,b, xy_rot = 15, azi_rot = 5):
    l0 = E_root(a,b)[0]
    t0 = E_root(a,b)[1]
    l_list = np.linspace(-0.8,0.8,100)
    t_list = np.linspace(-0.8,0.8,100)
    l_plot, t_plot = np.meshgrid(l_list, t_list)
    E_dens = E(l_plot, t_plot, a,b)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(t_plot, l_plot,  E_dens, 400, cmap='binary')
    ax.set_xlabel('\u03BB')
    ax.set_ylabel('t')
    ax.set_zlabel('Energy Density')
    ax.set_title('Energy density of α={} and β={} over a range of expectation values λ and t'.format(a,b))
    ax.view_init(xy_rot, azi_rot)

### Dispersion
def dispersion(l,t,a,b,k):
    return (np.sqrt((1 - 2*(1 + a) *t)**2 + (b * (1 + 2 *(1 + a) * l))**2 + 2 *(1 - 2 *(1 + a)* t)* (b *(1 + 2 *(1 + a)*l)) * np.cos(k)))

### returns the gap/(delta) value
def gap(l,t,a,b,k=np.pi):
    return 2*dispersion(l,t,a,b,k)

## plotting the main Phase Diagram of alpha against beta
def PhaseDiagramData(datapoints):
    amin = -0.99; 
    amax = -0.01; 
    bmin = 0.01; 
    bmax = 1.0; 
    alist = np.linspace(amin,amax,datapoints)
    blist = np.linspace(bmin,bmax,datapoints)
    a_plot = []
    b_plot = []
    
    ## *important note*: the loop below is computationally expensive and needs more runtime for datapoints larger than 200.
    ## it is more convenient to save the data in a separate csv file
    ## it is better to do retrieval and plotting separately to make adjusting the plot easier
    
    with open("gapdata_{}_corr.csv".format(datapoints), "w") as f:
        writegap = csv. writer(f)
        for a in alist:
            for b in blist:
                l0, t0 = E_root(a,b)
                delta = gap(l0,t0,a,b)
                writegap.writerow([delta])
                
                
## returns the index of the minimum gap value in the data              
def getMinimum(data): # data is square array
    indexlist = []
    for row in data:
        minval = min(row)
        mindex = int(np.where(row == minval)[0])
        indexlist.append(mindex)

    return indexlist


## finding the midpoint between the two beta values when there is an increase and decrease in values of delta
## i.e, the left(-1) and right(+1) indicies of the minimum index of the gap data
## then finding their corresponding midpoints of beta by list indexing
def midPoint(data, datapoints):
    bmin = 0.01; 
    bmax = 1.0; 
    blist = np.linspace(bmin,bmax,datapoints)
    minlist = getMinimum(data)
    midptlist = []
    idx = 0
    ## to override index out of range error
    for mindex in minlist:
        if len(blist) == minlist[idx]+1:
            beta_r = 1
            beta_l = blist[minlist[idx]-1]
            
        elif minlist[idx] == 0:
            beta_l = 0
            beta_r = blist[minlist[idx]+1]
            
        else:
            beta_l = blist[minlist[idx]-1]
            beta_r = blist[minlist[idx]+1]
            
        beta_mid = (beta_l + beta_r)/2
        midptlist.append(beta_mid)
        idx+=1
    return midptlist

## plotting the main phase diagram
def PlotPhaseDiagram(filename, datapoints, mode = "scatter"): ## make sure the number of data points of the filename and argument "datapoints" agree
    with open(filename) as f:
        deltalist = []
        for pt in f:
            if pt == '\n':
                continue
            else:
                deltalist.append(float(pt))
    deltamatrix = np.array(deltalist).reshape(datapoints, datapoints) ## transforming into a square matrix
    
    amin = -0.99; 
    amax = -0.01; 
    a_data = np.linspace(amin,amax,datapoints)
    b_data = midPoint(deltamatrix, datapoints)
    
    if mode =="scatter":
        plt.scatter(b_data, a_data, marker = "x")
    elif mode =="plot":
        plt.plot(b_data, a_data,marker=".", markersize=10)
    plt.xlim(0, 1);
    plt.ylim(-1, 0);
    plt.xlabel("\u03B2"); plt.ylabel("\u03B1")
    plt.title("Phase diagram")
    plt.show()
    
def Main_PhasePlot():
    filename = "gapdata_200_corr.csv"
    PlotPhaseDiagram(filename, 200, mode = "scatter")
    
if "__name__" == "__main__":
    Main_PhasePlot()
    PlotEnergy(a,b, xy_rot = 15, azi_rot = 5)
    