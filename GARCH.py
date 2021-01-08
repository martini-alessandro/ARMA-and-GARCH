#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:26:25 2021

Class that simulates a GARCH(p,q) process, and use it su estimate and forecast
volatility for stocks data 
"""
import numpy as np 

class GARCH(object): 
    
    def __init__(self, omega: float, alphas: np.ndarray, betas: np.ndarray, conditional_distribution):
        """Initiate GARcH(p, q) class, with p the lengths of the alpha 
        coefficients and q the legth of the beta coefficients. 
        Conditional distribution is the distribution of the random, independent
        increments, it generally has 0 mean
        """ 
        self._raiseInitError(omega, alphas, betas, conditional_distribution)
        self.alphas = alphas 
        self.betas = betas 
        self.omega = omega
        self.conditionalDistribution = conditional_distribution
        self.variance = self._unconditionalVariance(omega, alphas, betas)
        
        
    def simulate(self, length = 253, mean = 0): 
        return self._generateSimulation(length, mean)

    
    
    
    def _sample(self, mean, variance, size = 1, distribution = 'normal'): 
        """ Samples conditional distribution"""
        return np.random.normal(0, np.sqrt(variance), size = size)
    
    
    
    def _addNewElements(self, returns, volatilities, p, q, mean):
        """This generate new returns and volatility and return the updated 
        lists, from length N to length N + 1""" 
        #Update Variables 
        new_vol = self.omega\
                 + self.alphas[::-1][-p : ] @ np.array(returns)[-p : ] ** 2\
                 +  self.betas[::-1][-q : ] @ np.array(volatilities)[-q : ]
        new_rets = np.random.normal(mean, np.sqrt(new_vol))
        
        #Update Arrays 
        returns.append(new_rets)
        volatilities.append(new_vol)
        return returns, volatilities    
    
    
    def _generateSimulation(self, length, mean): 
        """Generate the simulation of GARCH(p,q) process after the Burn In 
        initial values. 10% more of returns are generated in order to make
        the result stable""" 
        
        p, q = len(self.alphas), len(self.betas)
        returns, volatilities = self._generateBurnIn(mean) 
        
        for _ in range(int(1.1 * length)):
            returns, volatilities = self._addNewElements(returns, volatilities,
                                                         p, q, mean)
            
        return np.array(returns)[-length : ], np.array(volatilities)[- length : ]
        
        
    def _generateBurnIn(self, mean):
       """
       Generate p = len(alphas) returns and q = len(betas) variances to initialize
       GARCH simulation, in order to have max(p, q) returns and values for variance 
       """       
       #Initialize variables 
       p, q = len(self.alphas), len(self.betas)
       volatilities = [self.variance]
       returns = [np.random.normal(mean, np.sqrt(self.variance))]
       
       #Generate new returns and volatilities since we have enough elements
       while len(returns) < max(p,q): 
           _p, _q = min(len(returns), p), min(len(volatilities), q)
           returns, volatilities = self._addNewElements(returns, volatilities, 
                                                        _p, _q, mean) 
       return returns, volatilities
   
   
    
    def _unconditionalVariance(self, omega, alphas, betas): 
        "Computes the asymptotic variance of the model"
        return omega / (1 - (alphas.sum() + betas.sum()))
        
    
    def _raiseInitError(self, omega, alphas, betas, conditional_distribution): 
        """
        Raises error if init variables are ill defined  

        """
        #List containing all the possible conditional distribution for returns
        availableDistributions = ['normal']
        
        #Raise Error in case of type error or not available conditional distribution
        if type(omega) != float and type(omega) != int:
            raise TypeError('Omega has to be float or int')
            
        if type(alphas) != np.ndarray or type(betas) != np.ndarray: 
            raise TypeError('Type of alphas and betas coefficients must be np.ndarray')
        
        if conditional_distribution.lower() not in availableDistributions: 
            raise ValueError('Conditional distirbution not available. The \
                             available conditional distributions are:\n{}'
                             .format('; '.join(availableDistributions)))
        return None
