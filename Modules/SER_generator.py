#This is the function generating the normally distributed area-specific emission rates (SERs) of BTEX for the specified stationery.

import torch

def SER_generator(condition, stationery, time): 
    stressors = condition[stationery].keys()

    #Converting the unit from [ng/cm2/min] -> [mg/cm2/s].
    means = torch.tensor([condition[stationery][stressor][0] for stressor in stressors]).reshape(1, -1) / 1000000 / 60   #shape: (1 <stationery>, 6 <stressors>).
    stds = torch.tensor([condition[stationery][stressor][1] for stressor in stressors]).reshape(1, -1) / 1000000 / 60

    #Randomly generate the SERs of BTEX for the stationery as tensor (shape: (1 <stationery>, 6 <stressors>)).
    stressors_SER = torch.maximum(torch.tensor(0.0), torch.normal(means, stds)) #shape: (1, 6).
    stressors_SER = torch.tile(stressors_SER, (time, 1))   #shape: (2400, 6).

    return stressors_SER