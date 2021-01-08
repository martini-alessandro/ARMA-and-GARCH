#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of GARCH(1,1) process for 10 years (253 trading days * 10 years)
"""

from GARCH import GARCH
import numpy as np
import matplotlib.pyplot as plt 

def plotSimulation(returns, variances): 
    fig, ax = plt.subplots(2)
    ax[0].plot(returns)
    ax[1].plot(np.sqrt(variances))
    ax[0].set_ylabel('Returns')
    ax[1].set_ylabel('Volatility')
    ax[1].set_xlabel('Time [a.u.]')
    return None 
    
    
if __name__ == '__main__':
    omega, alpha, beta = 3.6e-6, np.array([.1]), np.array([.88])
    g = GARCH(omega, alpha, beta, 'normal')
    returns, variances = g.simulate(2530)
    plotSimulation(returns, variances)
    