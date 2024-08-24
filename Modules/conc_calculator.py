#This is the function to calculate the BTEX emission concentrations from the stationery according to sequences of emission area.
import torch

def conc_calculator(seq_area, # A tensor indicating the sequence of emission area associated to the chosen stationery (shape: (2400)).
                    SER, #[Unit: mg/cm2/s], Shape: (2400, 6).
                    VB, #Volumn of the breathing zone [unit: m3].
                    AER, #Air exchange rate (AER) of the breathing zone [unit: h^-1].
                    dt = 1 #Δt, default as 1 [unit: s].
                    ):
    
    time = len(seq_area)

    #The unit of the input air exchange rates (AER) is [h^-1], which has to be converted to [s^-1] here for unit consistence with seq_area.
    AER_s = torch.tensor(AER / 3600)

    #Define a matrix of 'exp(-λ)'（i.e., e_factor） terms.
    e_factor = torch.exp(-AER_s) * dt

    i, j = torch.triu_indices(time, time)
    e_matrix = torch.zeros((time, time))
    e_matrix[i, j] = e_factor ** (j - i)
    e_matrix = e_matrix[:, :-1] #This is an upper triangular matrix with shape: (2400 <xi terms (defined as below)>, 2399 <concentration terms (expect c(0)>). 

    #Calculate the sequences of cumulative exposure concentration series [unit: mg/m3].
    steady_emission = (seq_area.reshape(-1, 1) * SER) / (VB * AER_s)   #Tensor, shape: (2400 <timestep>, 6 <stressors>).
    xi = (1 - e_factor) * steady_emission #Shape: (2400 <xi term for each timestep>, 6 <stressors>).
    xi_ = xi[:, :, None] #Expand xi to a 3D tensor (shape：(2400 <xi terms>, 6 <stressors>, 1 <concentration terms>)).
    e_matrix_= e_matrix[:, None, :] #Expand e_matrix to a 3D tensor (shape: (2400 <xi terms>, 1 <stressor>, 2399 <concentration terms (expect c(0)>)).
    xi_mtx_stressors = xi_ * e_matrix_ #Shape: (2400, 6, 2399)
    concentrations = xi_mtx_stressors.sum(dim = 0).transpose(0, 1) #Concentrations[1: time] = summing up all xi terms of each stressor along with timestep (shape: (2399, 6))
    concentrations = torch.vstack([torch.zeros((1, concentrations.shape[1])), concentrations]) #Fill the first row (t = 0) with 0 to get the intact concentrations[0: time]

    return concentrations #[Unit: mg/m3]