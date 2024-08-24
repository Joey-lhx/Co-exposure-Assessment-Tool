import torch
import pandas as pd

from src.students_const import StudentData
from src.speed_const import spread_speed
from src.area_const import placing_area

from Modules.SER_generator import SER_generator
from Modules.conc_calculator import conc_calculator

class student(object):
    def __init__(self, age, gender, time = 2400, period = 6, frequency = 180):

        #Define the physicological parameters of the student
        self.age = age
        self.gender = gender
        self.InhR = StudentData[age][gender]['InhR']
        self.weight = StudentData[age][gender]['Weight']

        #Define the stationery using states
        self.states = ['N', 'Pen', 'Correction Fluid', 'Marker', 'Pencil', 'Eraser']

        #Define the spreading speed [unit: cm2/s] and placing area [unit: cm2] for Type I, II stationery
        #Type I stationery spreading speed
        self.speed_Pen = spread_speed['Pen']
        self.speed_Correction = spread_speed['Correction']
        self.speed_Marker = spread_speed['Marker']

        #Type II stationery placing area
        self.area_Pencil = placing_area['Pencil']
        self.area_Eraser = placing_area['Eraser']
        
        #Define the time limt of a class and how many classes in a day
        self.time = time #[unit: s]
        self.period = period
        self.F = frequency  # The number of days the student go to school per year [days/year].

        #Define the stationery sequences (col: stationery, row: timestep. 0: no used, 1: used)
        #Type I stationery: cumulative using sequences (for calculating the sequence of cumulative emission area)
        self.seq_Pen_cum_using = None
        self.seq_Correction_cum_using = None
        self.seq_Marker_cum_using = None

        #Type II stationery: using sequences (the sequences of emission area would not change with time)
        self.seq_Pencil_using = None
        self.seq_Eraser_using = None

        #Define the area sequence of each stationery [unit: cm2]
        self.seq_Pen_area = None
        self.seq_Correction_area = None
        self.seq_Marker_area = None
        self.seq_Pencil_area = None
        self.seq_Eraser_area = None
        
        #Define the SERs [unit: mg/cm2/s] and exposure concentrations [mg/m3] of BTEX emitted from each stationery
        self.seq_Pen_SER = None
        self.seq_Correction_SER = None
        self.seq_Marker_SER = None
        self.seq_Pencil_SER = None
        self.seq_Eraser_SER = None

        self.seq_Pen_conc = None
        self.seq_Correction_conc = None
        self.seq_Marker_conc = None
        self.seq_Pencil_conc = None
        self.seq_Eraser_conc = None
        self.seq_Tot_conc = None

        #Define the inhalation doses [unit: mg/kg]
        self.seq_Pen_dose = None
        self.seq_Correction_dose = None
        self.seq_Marker_dose = None
        self.seq_Pencil_dose = None
        self.seq_Eraser_dose = None
        self.seq_Tot_dose = None

        #Define HQs and HI (non-cancerogenic risk)
        self.seq_Pen_HQ = None
        self.seq_Correction_HQ = None
        self.seq_Marker_HQ = None
        self.seq_Pencil_HQ = None
        self.seq_Eraser_HQ = None
        self.seq_Tot_HQ = None
        self.seq_HI = None

        #Define the cancerogenic risk
        self.seq_Pen_CR = None
        self.seq_Correction_CR = None
        self.seq_Marker_CR = None
        self.seq_Pencil_CR = None
        self.seq_Eraser_CR = None
        self.seq_Tot_CR = None
        self.seq_CR = None

    #Calculate the using sequence of each stationery
    def using_seq(self, stationery_using_seq):
        
        #Convert the string dateset (N, N, Pen, Marker, ...) of staionery using into boolean tensor (timesteps, stationery)
        #Initialize the whole stationery using sequence as 0
        # temporal_using_tensor = torch.zeros((self.time, len(self.states)), device='cuda' if torch.cuda.is_available() else 'cpu') #col:stationery, row: timestep
        temporal_using_tensor = torch.zeros((self.time, len(self.states)))
        state_index_map = {state: idx for idx, state in enumerate(self.states)}

        #set 1 to the specified row (time) and col (stationery)
        for t_, state in enumerate(stationery_using_seq):
            state_idx = state_index_map[state]
            temporal_using_tensor[t_, state_idx] = 1
        
        #Calculate the cumulative using (time) sequence
        #Type I (Using emission stationery): Pen, Correction Fluid, Marker
        Pen_idx = state_index_map['Pen']
        Correction_idx = state_index_map['Correction Fluid']
        Marker_idx = state_index_map['Marker']

        self.seq_Pen_cum_using = temporal_using_tensor[:, Pen_idx].cumsum(dim = 0)
        self.seq_Correction_cum_using = temporal_using_tensor[:, Correction_idx].cumsum(dim = 0)
        self.seq_Marker_cum_using = temporal_using_tensor[:, Marker_idx].cumsum(dim = 0)

        #Type II (Placing emission stationery): Pencil & rulers, Eraser
        Pencil_idx = state_index_map['Pencil']
        Eraser_idx = state_index_map['Eraser']

        self.seq_Pencil_using = temporal_using_tensor[:, Pencil_idx].clone()
        self.seq_Eraser_using = temporal_using_tensor[:, Eraser_idx].clone()

        #Find the index of the first non-zero element, and gives 1 to the elements following the fisrt non-zero element
        #Note: we do not calculate the cumulative time of 'placing-emission stationery' using, because their emission area would not change with time
        if torch.any(self.seq_Pencil_using != 0):
            Pencil_non_zero_idx = (self.seq_Pencil_using != 0).nonzero(as_tuple = True)[0].min().item()
            self.seq_Pencil_using[Pencil_non_zero_idx:] = 1
        else:
            self.seq_Pencil_using[:] = 0

        if torch.any(self.seq_Eraser_using != 0):
            Eraser_non_zero_idx = (self.seq_Eraser_using != 0).nonzero(as_tuple = True)[0].min().item()
            self.seq_Eraser_using[Eraser_non_zero_idx:] = 1
        else:
            self.seq_Eraser_using[:] = 0
    
    #Calculate the sequences of cumulative emission area [unit: cm2]
    def emission_area(self):
        
        #Type I stationery: sequences of cumulative emission area
        self.seq_Pen_area = self.seq_Pen_cum_using * self.speed_Pen
        self.seq_Correction_area = self.seq_Correction_cum_using * self.speed_Correction
        self.seq_Marker_area = self.seq_Marker_cum_using * self.speed_Marker

        #Type II stationery emission area
        self.seq_Pencil_area = self.seq_Pencil_using * self.area_Pencil
        self.seq_Eraser_area = self.seq_Eraser_using * self.area_Eraser

    #Calculate the exposure concentrations of BTEX
    def expo_conc(self, condition, VB, AER):

        #Generate SERs [unit: mg/cm2/s] firstly.
        self.seq_Pen_SER = SER_generator(condition = condition, stationery = 'Pen', time = self.time)
        self.seq_Correction_SER = SER_generator(condition = condition, stationery = 'Correction', time = self.time)
        self.seq_Marker_SER = SER_generator(condition, stationery = 'Marker', time = self.time)
        self.seq_Pencil_SER = SER_generator(condition, stationery = 'Pencil', time = self.time)
        self.seq_Eraser_SER = SER_generator(condition, stationery = 'Eraser', time = self.time)

        #Calculate exposure concentratrions of breathing zone [unit: mg/m3]
        self.seq_Pen_conc = conc_calculator(self.seq_Pen_area, self.seq_Pen_SER, VB, AER)
        self.seq_Correction_conc = conc_calculator(self.seq_Correction_area, self.seq_Correction_SER, VB, AER)
        self.seq_Marker_conc = conc_calculator(self.seq_Marker_area, self.seq_Marker_SER, VB, AER)
        self.seq_Pencil_conc = conc_calculator(self.seq_Pencil_area, self.seq_Pencil_SER, VB, AER)
        self.seq_Eraser_conc = conc_calculator(self.seq_Eraser_area, self.seq_Eraser_SER, VB, AER)
        self.seq_Tot_conc = self.seq_Pen_conc + self.seq_Correction_conc + self.seq_Marker_conc + self.seq_Pencil_conc + self.seq_Eraser_conc
    
    #Calculate the sequences of BTEX inhalation doses [unit: mg/kg]
    def expo_dose(self):

        #Aggregate the inhaled dose of each timestep to get the cumulative sequences of inhalation doses
        self.seq_Pen_dose = (self.seq_Pen_conc * self.InhR).cumsum(dim = 0) * self.period * self.F / (self.weight * 365)
        self.seq_Correction_dose = (self.seq_Correction_conc * self.InhR).cumsum(dim = 0) * self.period * self.F / (self.weight * 365)
        self.seq_Marker_dose = (self.seq_Marker_conc * self.InhR).cumsum(dim = 0) * self.period * self.F / (self.weight * 365)
        self.seq_Pencil_dose = (self.seq_Pencil_conc * self.InhR).cumsum(dim = 0) * self.period * self.F / (self.weight * 365)
        self.seq_Eraser_dose = (self.seq_Eraser_conc * self.InhR).cumsum(dim = 0) * self.period * self.F / (self.weight * 365)
        self.seq_Tot_dose = self.seq_Pen_dose + self.seq_Correction_dose + self.seq_Marker_dose + self.seq_Pencil_dose + self.seq_Eraser_dose
    
    #Calculate the non-cancerogenic risk (HQ) by dividing inhalation doses with reference dose of each stressor (RfD)
    def HQ(self, RfD): #RfD is a dictionary like {'Benzene': 0.002, 'Toluene': 0.08, ...}

        #The RfD used here should be converted to a (6) tensor which be used as divisors for dose tensor (2400, 6) by broadcasting.
        #Note: the order stressors in RfD should be consisted with the that in the dose sequence
        tensor_RfD = torch.tensor(list(RfD.values()))

        #Calculate the HQ sequence of each stressor from each stationery
        self.seq_Pen_HQ = self.seq_Pen_dose.div(tensor_RfD)
        self.seq_Correction_HQ = self.seq_Correction_dose.div(tensor_RfD)
        self.seq_Marker_HQ = self.seq_Marker_dose.div(tensor_RfD)
        self.seq_Pencil_HQ = self.seq_Pencil_dose.div(tensor_RfD)
        self.seq_Eraser_HQ = self.seq_Eraser_dose.div(tensor_RfD)

        #Calculate the sequence of the total HQ for each stressor from all stationery.
        self.seq_Tot_HQ = self.seq_Pen_HQ + self.seq_Correction_HQ + self.seq_Marker_HQ + self.seq_Pencil_HQ + self.seq_Eraser_HQ

        #Calculate the sequence of hazard index (HI) by summing up HQs caused by all stressors.
        self.seq_HI = self.seq_Tot_HQ.sum(dim = 1)
    
    #Calculate the cancerogenic risk (CR) by multiplying slope factors (SF).
    def CR(self, SF):

        #The SF used here should be converted to a (6) tensor which be used as multiplier for dose tensor (2400, 6) by broadcasting.
        tensor_SF = torch.tensor(list(SF.values()))

        #Calculate the sequence of CR sequence of each stressor from each stationery (assuming the lifetime of individual is 70 years).
        self.seq_Pen_CR = (self.seq_Pen_dose / 70) * tensor_SF
        self.seq_Correction_CR = (self.seq_Correction_dose / 70) * tensor_SF
        self.seq_Marker_CR = (self.seq_Marker_dose / 70) * tensor_SF
        self.seq_Pencil_CR = (self.seq_Pencil_dose / 70) * tensor_SF
        self.seq_Eraser_CR = (self.seq_Eraser_dose / 70) * tensor_SF

        #Calculate the sequence of the total CR for each stressor from all stationery.
        self.seq_Tot_CR = self.seq_Pen_CR + self.seq_Correction_CR + self.seq_Marker_CR + self.seq_Pencil_CR + self.seq_Eraser_CR

        #Calculate the sequence of the eventually CR by summing up CRs caused by all stressor
        self.seq_CR = self.seq_Tot_CR.sum(dim = 1)