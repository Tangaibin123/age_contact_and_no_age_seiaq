import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt

#总人数
N=10000
#传染系数
beta =0.6

gamma=0.1
Te = 14
I_0=2
E_0=0
R_0=0
Q_0=0
D_0=0
S_0 = N-I_0-E_0-R_0
T = 150

INI = (S_0,E_0,I_0,R_0)

def funcSIR(inivalue,_):
    Y = np.zeros(4)
    X = inivalue
    Y[0]

def funcSEIR