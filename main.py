#Simulating and analyzing the exposure risks for different student groups.
import time
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import gc

from src import students_const
from src import SER_const
from src.RfD_dict import RfD
from src.SF_dict import SF

import InferExpo
import Analyzer


#Record the running time.
start_time = time.time()

#Get the the path of this file
script_directory = os.path.dirname(os.path.abspath(__file__))

#Separate all student groups into A (7 ~ 9 years old) and B (10 ~ 12 years old).
group_A = {i: students_const.StudentData[i] for i in range(7, 10)}
group_B = {i: students_const.StudentData[i] for i in range(10, 13)}

group_list = [group_A, group_B]
condition_list = [SER_const.condition_1, SER_const.condition_2, SER_const.condition_3, SER_const.condition_4]

#Iterate A and B groups (2).
for group in group_list:
    if group == group_A:
        syn_sample = pd.read_csv("Dataset//syn_sample_A.csv")
    else:
        syn_sample = pd.read_csv("Dataset//syn_sample_B.csv")
    
    #Iterate students subgroups in the group (3).
    for age, gender in group.items():
        student_age = age
        
        #Iterate the gender (2).
        for gender_i, info in gender.items():
            student_gender = gender_i

            #Iterate SERs conditions (4).
            for condition in condition_list:
                if condition is SER_const.condition_1:
                    condition_idx = 'C1'
                elif condition is SER_const.condition_2:
                    condition_idx = 'C2'
                elif condition is SER_const.condition_3:
                    condition_idx = 'C3'
                elif condition is SER_const.condition_4:
                    condition_idx = 'C4'

                #print the current process
                print(f"Processing {student_gender}_{student_age} at {condition_idx}")

                #Inference the exposure process
                student_expo = InferExpo.process_samples(syn_data = syn_sample, #you can adjust 0 ~ 10000 here to set the sample size of simulation (e.g., syn_sample[0: 100])
                                                         age = student_age, gender = student_gender, 
                                                         time = 2400, period = 6, frequency = 180,
                                                         condition = condition, VB = 0.3, AER = 0.75,
                                                         RfD = RfD, SF = SF)
                
                #Analyze exposure risks
                #Set the prefix of the output files
                filename = f'{student_gender}_{student_age}_{condition_idx}.csv'

                #Set the paths of output files
                conc_path = script_directory + "/Results/Risks/Conc/" + filename
                risks_path = script_directory + "/Results/Risks/Risks_range/" + filename
                CTH_path = script_directory + "/Results/Risks/CTH/" + filename
                HICR_path = script_directory + "/Results/Contributions/HICR/abs/" + filename
                HICR_norm_path = script_directory + "/Results/Contributions/HICR/" + filename
                S2B_path = script_directory + "/Results/Contributions/stationery_BTEX/abs/" + filename
                S2B_norm_path = script_directory + "/Results/Contributions/stationery_BTEX/" + filename

                #Start the analysis
                Analyzer.risks_analyzer(student_expo, time = 2400, period = 6, 
                                        conc_path = conc_path, risks_path = risks_path, 
                                        CTH_path = CTH_path, 
                                        HICR_path = HICR_path, HICR_norm_path = HICR_norm_path, 
                                        S2B_path = S2B_path, S2B_norm_path = S2B_norm_path)
                
            #Clean the cache
            print("Cleaning cache...")
            del student_expo
            gc.collect()

end_time = time.time()
elapsed_time = (end_time - start_time) / 3600
print(f'Totally cost {elapsed_time} hours.')