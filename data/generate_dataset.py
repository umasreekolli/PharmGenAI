import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

drugs = ['Warfarin', 'Paracetamol', 'Ibuprofen', 'Fluconazole',
         'Aspirin', 'Metformin', 'Atorvastatin', 'Omeprazole',
         'Codeine', 'Clopidogrel']

cyp2d6_opts  = ['Poor', 'Intermediate', 'Normal', 'Ultrarapid']
cyp3a4_opts  = ['Poor', 'Normal', 'Fast']
cyp2c19_opts = ['Poor', 'Normal', 'Rapid']

cyp2d6_risk  = {'Poor': 3, 'Intermediate': 2, 'Normal': 1, 'Ultrarapid': 2}
cyp3a4_risk  = {'Poor': 3, 'Normal': 1, 'Fast': 1.5}
cyp2c19_risk = {'Poor': 3, 'Normal': 1, 'Rapid': 1.5}

drug_cyp_weight = {
    'Warfarin':     {'CYP2D6': 0.2, 'CYP3A4': 0.3, 'CYP2C19': 0.5},
    'Paracetamol':  {'CYP2D6': 0.3, 'CYP3A4': 0.5, 'CYP2C19': 0.2},
    'Ibuprofen':    {'CYP2D6': 0.3, 'CYP3A4': 0.5, 'CYP2C19': 0.2},
    'Fluconazole':  {'CYP2D6': 0.2, 'CYP3A4': 0.6, 'CYP2C19': 0.2},
    'Aspirin':      {'CYP2D6': 0.3, 'CYP3A4': 0.4, 'CYP2C19': 0.3},
    'Metformin':    {'CYP2D6': 0.2, 'CYP3A4': 0.4, 'CYP2C19': 0.4},
    'Atorvastatin': {'CYP2D6': 0.1, 'CYP3A4': 0.8, 'CYP2C19': 0.1},
    'Omeprazole':   {'CYP2D6': 0.2, 'CYP3A4': 0.3, 'CYP2C19': 0.5},
    'Codeine':      {'CYP2D6': 0.8, 'CYP3A4': 0.1, 'CYP2C19': 0.1},
    'Clopidogrel':  {'CYP2D6': 0.1, 'CYP3A4': 0.3, 'CYP2C19': 0.6},
}

records = []

for _ in range(n):
    drug    = np.random.choice(drugs)
    age     = np.random.randint(18, 80)
    gender  = np.random.choice(['M', 'F'])   # FIXED (more realistic)
    c2d6    = np.random.choice(cyp2d6_opts)
    c3a4    = np.random.choice(cyp3a4_opts)
    c2c19   = np.random.choice(cyp2c19_opts)
    comorbid   = np.random.randint(0, 4)
    num_drugs  = np.random.randint(1, 6)

    # Simulated DDI risk (important upgrade)
    ddi_risk = 0
    if num_drugs > 1 and np.random.rand() < 0.3:
        ddi_risk = 1.5

    w = drug_cyp_weight[drug]

    risk_score = (
        w['CYP2D6']  * cyp2d6_risk[c2d6] +
        w['CYP3A4']  * cyp3a4_risk[c3a4] +
        w['CYP2C19'] * cyp2c19_risk[c2c19] +
        (age / 80) * 0.5 +
        comorbid * 0.3 +
        num_drugs * 0.2 +
        ddi_risk +
        np.random.normal(0, 0.3)
    )

    if risk_score < 1.8:
        label = 0  # Low
    elif risk_score < 2.8:
        label = 1  # Medium
    else:
        label = 2  # High

    records.append([
        drug, age, gender, c2d6, c3a4, c2c19,
        comorbid, num_drugs, ddi_risk, label
    ])

df = pd.DataFrame(records, columns=[
    'Drug', 'Age', 'Gender', 'CYP2D6', 'CYP3A4', 'CYP2C19',
    'Comorbidities', 'NumDrugs', 'DDI_Risk', 'ADR_Risk'
])

# FIXED PATH (VERY IMPORTANT)
df.to_csv('adr_dataset.csv', index=False)

print("Done! Dataset shape:", df.shape)
print(df['ADR_Risk'].value_counts())