from tqdm import tqdm
import pandas as pd
from Modules.student import student
from src.SER_const import *
# from src.RfD_dict import RfD
# from src.SF_dict import SF


def process_samples(syn_data, 
                    age, gender, 
                    time, period, frequency, 
                    condition, VB, AER,
                    RfD, SF):
    
    students = []

    for idx in tqdm(range(len(syn_data)), desc = "Processing samples"):
        student_behavior_seq = syn_data.iloc[idx]
        student_profile = student(age, gender, time, period, frequency)
        student_profile.using_seq(student_behavior_seq)
        student_profile.emission_area()
        student_profile.expo_conc(condition, VB, AER)
        student_profile.expo_dose()
        student_profile.HQ(RfD)
        student_profile.CR(SF)

        students.append(student_profile)
    
    return students

#Example
# Syn_sample_A = pd.read_csv("Dataset//syn_sample_A.csv")

# C1_Boy_7 = process_samples(syn_data = Syn_sample_A[0: 10], 
#                            age = 7, gender = 'Boy', 
#                            time = 2400, period = 6, frequency = 180,
#                            condition = condition_1, VB = 0.3, AER = 0.75,
#                            RfD = RfD, SF = SF)

# print("HI is: ", C1_Boy_7[0].seq_HI[-1])
# print("CR is: ", C1_Boy_7[0].seq_CR[-1])