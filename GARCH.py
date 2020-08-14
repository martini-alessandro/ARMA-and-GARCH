# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 17:27:02 2020

@author: Workplace
"""
import numpy as np
import sphinx

class GARCH(object): 
    """Create a GARCH object that simulates a GARCH(p, q) process with 
        increments extracted from a given distribution, default is normal 
        distribution. 
        GARCH model is given by 
        .. math::
        \\sigma^2_t = mean + \sum_i \alphas_i x^2(t-1) + \sum_i \beta_i sigma^2(t-1)
        p and q parameters have to be numpy arrays""" 
        
    def __init__(self, variance, mean, alphas, betas, distribution = 'normal'): 
        
        self.variance = np.array([variance])
        self.mean = mean
        self.distribution = distribution
        self.returns = self._extractReturn()
        self.alphas = alphas 
        self.betas = betas
        
    def _extractReturn(self, variance = None): 
        """Initialize the first value of returns for GARCH models with 
        given distribution. If no value is passed, compute the extraction 
        using the current value of conditional variance""" 
        #Compute with current variance value if no value inserted
        if variance == None:
            v = self.variance[-1]
            
        #Compute with inserted value for the variance
        elif type(variance) == int or type(variance) == float: 
            v = variance
        
        #Raise Error in other cases
        else:
            raise ValueError('Variance parameter has to be None, int or float')
        
        #Extract from normal distribution 
        if self.distribution.lower() == 'normal':
                return np.array([np.sqrt(v) * np.random.normal(0, 1)])
        else:
            raise ValueError('Chosen distribution is not available')
            
    
    def simulate(self, size = 1000): 
        """Simulate a GARCH process of a given length"""
        
        #simulate untill datas and parameters have same length. Do not rec datas
            #Initialize variables
        rets = self.returns[::-1]
        var = self.variance[::-1]
        
            #Main Loop 
        while rets.size < self.alphas.size or var.size < self.betas.size:
            #Correct length to perform scalar product
            p, q = len(rets), len(var)
            #Update conditional variance and returns 
            newVariance = self.mean + self.alphas[: p] @ (rets ** 2) \
                + self.betas[:q] @ var
            newRet = self._extractReturn(variance = float(newVariance))
            #record values 
            var = np.insert(var, 0, newVariance)
            rets = np.insert(rets, 0, newRet)
            print(var, rets)
        
        for i in range(2 * size): 
            #Try to implement recursively
            p, q = self.alphas.size, self.betas.size
            #Update conditional variance and returns
            newVariance = self.mean + self.alphas @ (rets ** 2)[ : p] \
                + self.betas @ var[ : q]
            newRet = self._extractReturn(variance = float(newVariance))
            #Store values
            var = np.insert(var, 0, newVariance)
            rets = np.insert(rets, 0, newRet)
            #Update attributes
            self.returns = rets[ : size][::-1]
            self.variance = var[ : size][::-1]
            
        return self.returns, self.variance
            

        