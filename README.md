# Elementary School Students Exposure to Emission of BTEX from Scented Stationery


![Graphical abstract of the paper.](Graphical_Abstract.png)

Graphical abstract of the paper.

## **📮**News


💡 **Next Step:**  Organize and upload the stationery using behavior simulation script.

💡 **2024-08-24:** Release the codes for risks assessment framework, including risks estimation and analysis.

## ✒️ Configurations for running the codes

### 1. Conda Environment

We test our code on **Python 3.12.4** and **Pytorch 2.3.0** ([https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)).  The rest of packages were storaged in environment.yml.

```bash
conda env create -f environment.yml
conda activate Expo
```

**Alternative:** pip installation

### 2. Get data

There are two options for obtaining the simulation data of stationery usage behavior:

- **Using preset data**
    
    We provide two .csv files containing 10,000 synthetic samples of stationery usage sequences over 2,400 seconds (a 40-minute class) for group A and B students, respectively. The files are storaged in the \Dataset document.
    
- **Generate your own data**
    
    We will soon publish the script for the behavior simulation algorithm based on the Markov chain model described in the paper.
    

## 🏃🏻‍♂️Explanation for critical scripts
![The logic and usage of running criticals scripts.](Critical_Scripts.png)

The logic and usage of running criticals scripts.

- **\Modules**
    - **conc_calculator.py:** Estimates BTEX exposure concentrations in the breathing zone based on SERs data.
    - **SER_generator.py:** Generates random SERs according to measured data fluctuation trends to incorporate uncertainty.
    - **student.py:** Packages information such as effective emission area, ADDs, and Risks for each simulated student sample for risk statistics and analysis.
- **\src**: Contains files with constants including physiological parameters, SERs, stationery-related spreading speeds, and toxicological data.
- **\Results**: Stores risk assessment and analysis results, primarily used for creating data charts presented in the paper.
- **\main.py**: Combines InferExpo.py and Analyzer.py to assess and analyze exposure risks for each simulated student sample iteratively.

## **Acknowlegement**


We sincerely thank the open-sourcing of [**SCB-dataset**](https://github.com/Whiffe/SCB-dataset), which provides an opportunity for us to characterize students' stationery-using behavior.

## Misc

For further questions, please contact [Joey_li@xs.ustb.edu.cn](mailto:Joey_li@xs.ustb.edu.cn).
