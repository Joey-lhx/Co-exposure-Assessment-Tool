from tqdm import tqdm
import pandas as pd
import numpy as np

def risks_analyzer(student_profiles, 
                   time,
                   period,
                   conc_path,
                   risks_path,
                   CTH_path,
                   HICR_path,
                   HICR_norm_path,
                   S2B_path,
                   S2B_norm_path):
    
    #Define the stressors and stationery names.
    stressors = ['Benzene', 'Toluene', 'Ethylbenzene', 'O-xylene', 'M-xylene', 'P-xylene']
    stationery = ['Pen', 'Correction Fluid', 'Marker', 'Pencil & Rulers', 'Eraser']

    #Preparation for [part 1]: some lists for collecting overall risks data.
    list_HI_CR = [] #Record the HI and CR at the end of the day.
    list_seq_HI = [] #Record the HI sequences of all students ()
    list_conc = []

    #Preparation for [part 2]: the concentration ranges for doing the concentration - exposure duration statistics.
    #Be careful to set the ranges!
    range_1 = np.arange(0, 0.001, 0.0001)
    range_2 = np.arange(0.001, 0.01, 0.001)
    range_3 = np.arange(0.01, 0.22, 0.02)
    conc_ranges = np.concatenate((range_1, range_2, range_3))
    dict_conc_ranges = {conc_ranges[i + 1]: np.around(conc_ranges[i + 1],decimals = 4) for i in range(len(conc_ranges) - 1)}
    df_conc_duration = pd.DataFrame(columns = dict_conc_ranges.keys()) #Build up a DataFrame to store durations of each duration step for all students.

    #Preparation for [part 3]: 
    stationery_HI = 0
    stationery_CR = 0

    Pen_HQ = 0
    Correction_HQ = 0
    Marker_HQ = 0
    Pencil_HQ = 0
    Eraser_HQ = 0

    #Collecting the relavant data of each student.
    for student in tqdm(student_profiles, desc = "Analyzing risks and contributions"):

        #Part 1: Determine the average risks of overall students.
        student_HI_CR = [student.seq_HI[-1], student.seq_CR[-1]] #Student's HI and CR at the end of a day.
        list_HI_CR.append(student_HI_CR)

        student_conc_seq = student.seq_Tot_conc
        list_conc.append(student_conc_seq)

        student_seq_HI = student.seq_HI
        list_seq_HI.append(student_seq_HI)

        #Part 2: Determine the exposure duration under the specific concentration(ΣBTEX) levels.
        counts = {conc: 0 for conc in dict_conc_ranges.keys()}

        BTEX_concentrations = student.seq_Tot_conc.sum(dim = 1) #The summation of all stressors concentrations. (shape: (2400 <timestep>, 1 <sum of concentrations>)).
        for concentration in BTEX_concentrations:
            for range_idx, threshold in dict_conc_ranges.items():
                if concentration < threshold:
                    counts[range_idx] += 1

        df_conc_duration = df_conc_duration._append(counts, ignore_index = True)

        #Part 3: Contribution Analysis.
        #3.1 Stationery contribution.
        stationery_HI += pd.DataFrame([[student.seq_Pen_HQ[-1:].sum(dim = 1).item(), 
                                        student.seq_Correction_HQ[-1:].sum(dim = 1).item(),
                                        student.seq_Marker_HQ[-1:].sum(dim = 1).item(),
                                        student.seq_Pencil_HQ[-1:].sum(dim = 1).item(),
                                        student.seq_Eraser_HQ[-1:].sum(dim = 1).item()]], columns = stationery)
        
        stationery_CR += pd.DataFrame([[student.seq_Pen_CR[-1:].sum(dim = 1).item(), 
                                        student.seq_Correction_CR[-1:].sum(dim = 1).item(),
                                        student.seq_Marker_CR[-1:].sum(dim = 1).item(),
                                        student.seq_Pencil_CR[-1:].sum(dim = 1).item(),
                                        student.seq_Eraser_CR[-1:].sum(dim = 1).item()]], columns = stationery)
        
        #3.2 The composition of BTEX in each stationery (summing up the HI from each stationery of all students)
        Pen_HQ += student.seq_Pen_HQ[-1:]
        Correction_HQ += student.seq_Correction_HQ[-1:]
        Marker_HQ += student.seq_Marker_HQ[-1:]
        Pencil_HQ += student.seq_Pencil_HQ[-1:]
        Eraser_HQ += student.seq_Eraser_HQ[-1:]

    #Do statistics to these data
    #1. Concentrations sequences and risks overview
    df_conc = pd.DataFrame(np.mean(np.array(list_conc), axis = 0), columns = stressors) #The sequences of average concentrations at each timestep (shape: (2400 <timesteps>, 6 <stressors>)) 存疑
    df_HI_CR = pd.DataFrame(np.array(list_HI_CR), columns = ['HI', 'CR']) #The sequences of risks (shape:(10000 <students>, 2 <risks types>)).
    
    #2.1 Concentration sequences and HI
    df_seq_HI = pd.DataFrame(np.array(list_seq_HI)) #Shape: (10000 <students>, 2400 <timesteps>)
    df_conc_HI = pd.DataFrame(index = df_seq_HI.index, columns = df_conc_duration.columns) #Shape:(2 <risk types>, 28 <duration steps>).
    row_indices = np.arange(df_conc_duration.shape[0])[:, None] # The index of student.
    col_indices = df_conc_duration.values.astype(int) - 1 #The index of ΣBTEX concentration.
    df_conc_HI.values[:] = df_seq_HI.values[row_indices, col_indices] #Locate the HI of the student by concentration index.
    
    #2.2 Concentration, duration and HI (CTH)
    df_CTH = pd.DataFrame(columns = ['Conc', 'Duration', 'HI'])
    df_CTH['Conc'] = conc_ranges[1:]
    df_CTH['Duration'] = ((time - 1) - np.array((df_conc_duration - 1).mean(axis = 0))) * period / 3600
    df_CTH['HI'] = np.array(df_conc_HI.mean(axis = 0))

    #3.1 The stationery contributing to the HI and CR
    df_HICR = pd.concat([stationery_HI, stationery_CR], ignore_index = True) # The summation of HI and CR of each stationery associated to the student group, shape: (2 <risk type>, 5 <stationery>)
    df_HICR_norm = df_HICR.div(df_HICR.sum(axis = 1), axis = 0)  #Normalized values

    #3.2 The stressor - stationery contribution for student groups
    # df_stationery_BTEX = pd.concat([Pen_HQ, Correction_HQ, Marker_HQ, Pencil_HQ, Eraser_HQ], ignore_index = True)
    df_stationery_BTEX = pd.DataFrame([Pen_HQ.tolist()[0], Correction_HQ.tolist()[0], Marker_HQ.tolist()[0], Pencil_HQ.tolist()[0], Eraser_HQ.tolist()[0]], columns = stressors)  #[0] because tolist return a 2D list with shape of (1, 6)
    df_stationery_BTEX_norm = df_stationery_BTEX.div(df_stationery_BTEX.sum(axis = 0), axis = 1)


    #Output the results
    df_conc.to_csv(conc_path)
    df_HI_CR.to_csv(risks_path)
    df_CTH.to_csv(CTH_path)
    df_HICR.to_csv(HICR_path)
    df_HICR_norm.to_csv(HICR_norm_path)
    df_stationery_BTEX.to_csv(S2B_path)
    df_stationery_BTEX_norm.to_csv(S2B_norm_path)