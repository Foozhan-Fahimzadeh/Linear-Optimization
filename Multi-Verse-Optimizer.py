import numpy  as np
import math
import random
import os

def function(x):
    val = 4*x[0]**2 - 2.1*x[0]**4 + (1/3)*x[0]**6 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4
    return val

#Initialize Variables
def Univese(universes , min_values , max_values , function = function):
    cosmos = np.zeros((universes, len(min_values) + 1))
    for i in range(universes):
        for j in range(len(min_values)):
             cosmos[i,j] = random.uniform(min_values[j], max_values[j])
        cosmos[i,-1] = function(cosmos[i,0:cosmos.shape[1]-1])
    return cosmos

#Fitness
def fitness(cosmos): 
    fit_arr = np.zeros((cosmos.shape[0], 2))
    for i in range(fit_arr.shape[0]):
        fit_arr[i,0] = 1/(1+ cosmos[i,-1] + abs(cosmos[:,-1].min()))
    fit_sum = fit_arr[:,0].sum()
    fit_arr[0,1] = fit_arr[0,0]
    for i in range(1 , fit_arr.shape[0]):
        fit_arr[i,1] = (fit_arr[i,0] + fit_arr[i-1,1])
    for i in range(fit_arr.shape[0]):
        fit_arr[i,1] = fit_arr[i,1]/fit_sum
    return fit_arr

#Random Selection 
def roulette_wheel(fit_arr): 
    p = 0
    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    for i in range(0, fit_arr.shape[0]):
        if (random <= fit_arr[i, 1]):
          p = i
          break
    return p

#White Hole
def white_hole(cosmos, fit_arr, best_universe, wep , tdr , min_values, max_values, function = function):
    for i in range(cosmos.shape[0]):
        for j in range(len(min_values)):
            r1 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (r1 < fit_arr[i, 1]):
                white_hole_i = roulette_wheel(fit_arr)       
                cosmos[i,j] = cosmos[white_hole_i,j]       
            r2 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                       
            if (r2 < wep):
                r3 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)  
                if (r3 <= 0.5):   
                    r = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1) 
                    cosmos[i,j] = best_universe [j] + tdr*((max_values[j] - min_values[j])*r + min_values[j]) 
                elif (r3 > 0.5):  
                    r = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1) 
                    cosmos[i,j] = np.clip((best_universe [j] - tdr*((max_values[j] - min_values[j])*r + min_values[j])),min_values[j],max_values[j])
        cosmos[i, -1] = function(cosmos[i, 0:cosmos.shape[1]-1])
    return cosmos

#MVO
def mvo(universes , min_values , max_values , iterations = 100 , function = function):    
    count   = 0 
    cosmos  = Univese(universes,  min_values, max_values, function)
    fit_arr = fitness(cosmos)    
    best_universe = np.copy(cosmos[cosmos[:,-1].argsort()][0,:])
    wep_max = 1
    wep_min = 0.1   
    while (count <= iterations):        
        print(f"in iteration = {count} the value of function is equal to {best_universe[-1]}")             
        wep = wep_min + count*((wep_max - wep_min)/iterations)
        tdr = 1 - (math.pow(count,1/6)/math.pow(iterations,1/6))        
        cosmos = white_hole(cosmos, fit_arr , best_universe, wep, tdr, min_values, max_values, function = function)
        fit_arr = fitness(cosmos) 
        value = np.copy(cosmos[cosmos[:,-1].argsort()][0,:])
        if(best_universe[-1] > value[-1]):
            best_universe  = np.copy(value)        
        count = count + 1         
    print(f"the best universe is {best_universe}")    
    return best_universe 

#testing 


mvo = mvo(universes = 50, min_values = [-5,-5], max_values = [5,5], iterations = 100, function = function)

