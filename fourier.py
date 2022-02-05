# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 11:40:28 2022
@author: RobertoMelo
"""
from scipy.integrate import quad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importando Dados de Simulink em Arrays
WS = pd.read_excel('simulink_data.xlsx')
x_train = np.array(WS[['tout']])
y_train = np.array(WS[['freq']])

tout = []
freq = []
for i in range(x_train.size):
    tout.append(x_train[i][0])
    freq.append(y_train[i][0])
  

def fourier(tout, freq):
    T = 1
    t0 = 0
    w = 1
    fv = []
    for i in range(len(tout)):
        t = tout[i]
        y = freq[i]
        # Integral de f(t)|[0,T] =>T*F(t)
        ## a0 = (1/T)*sum(y*i for i in np.arange(t0, t0+T, 0.018))
        l1,l2 = quad(lambda i: y, t0, t0+T)
        a0 = (1/T)*(l1-l2)
        
        # (y*cos(w*t))' =>  y*w*-sin(w*t)
        ## a1 = (2/T)*sum( y*w*(-1)*np.sin(w*i) for i in np.arange(t0, t0+T, 0.018))
        m1,m2 = quad(lambda i: y*np.cos(w*i), t0, t0+T)
        a1 = (2/T)*(m1-m2)
        
        # (y*sin(w*t))' => y*w*(-cos(w*t))
        ##b1 = (2/T)*sum( y*w*(np.cos(w*i)) for i in np.arange(t0, t0+T, 0.018))
        n1,n2 = quad(lambda i: y*np.sin(w*i), t0, t0+T)
        b1 = (2/T)*(n1-n2)
        
        # Somatorio
        f = a0 + a1*np.cos(w*t) + b1*np.sin(w*t)
        fv.append(f)
    return fv

theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
k = np.trapz(2, x=np.sin(theta))
print(k)


# Plotting
fig, ax1 = plt.subplots()

ax1.set_xlabel('tempo (t)')
ax1.set_ylabel('Sinal+Ruido (w)',color='red')
ax1.plot(tout,freq, color='red')

ax2 = ax1.twinx()
ax2.set_ylabel('Pos-Fourier (w)', color='blue')
ax2.plot(tout,fourier(tout,freq), color='blue')

plt.title('Série Fourier Frequencias - Roberto Marcos')
plt.show()
