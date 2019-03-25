import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
df_ = df = pd.read_csv('./data/train_LCCS.csv', encoding='utf-8')
# df[['ID','Age at diagnosis', 'CS tumor size (2004+)', 'CS extension (2004+)', 'CS mets at dx (2004+)', 'Regional nodes examined (1988+)', 'Regional nodes positive (1988+)']] = df[['ID','Age at diagnosis', 'CS tumor size (2004+)', 'CS extension (2004+)', 'CS mets at dx (2004+)', 'Regional nodes examined (1988+)', 'Regional nodes positive (1988+)']].astype(object)
df[['Sex','Histologic Type ICD-O-3', 'Histologic Type 2', 'Grade', 'RX Summ--Scope Reg LN Sur (2003+)', 'CS extension (2004+)', 'CS mets at dx (2004+)', 'Stage', 'T8', 'N8', 'M8', 'Marital status at diagnosis', 'Lung - Pleural/Elastic Layer Invasion (PL) by H and E or Elastic Stain', 'Lung - Separate Tumor Nodules - Ipsilateral Lung','Lung - Surgery to Primary Site (1988-2015)', 'Lung - Surgery to Other Regional/Distant Sites (1998+)']] = df[['Sex','Histologic Type ICD-O-3', 'Histologic Type 2', 'Grade', 'RX Summ--Scope Reg LN Sur (2003+)', 'CS extension (2004+)', 'CS mets at dx (2004+)', 'Stage', 'T8', 'N8', 'M8', 'Marital status at diagnosis', 'Lung - Pleural/Elastic Layer Invasion (PL) by H and E or Elastic Stain', 'Lung - Separate Tumor Nodules - Ipsilateral Lung','Lung - Surgery to Primary Site (1988-2015)', 'Lung - Surgery to Other Regional/Distant Sites (1998+)']].astype(object)
print(df.info())
print(df.head())
label = df[['LCCS', 'Survival months']]
df = df.drop(['LCCS', 'Survival months'], axis=1)
label.to_csv('./data/train_LCCS_label.csv', encoding='utf-8', index=False)
cols = df.select_dtypes(include=[object]).columns
# cols = ['Sex', 'Histologic Type ICD-O-3', 'Histologic Type 2', 'Grade', 'RX Summ - Scope Reg LN Sur (2003+)', 'LCCS', 'Survival months', 'OS', 'Race recode', 'Marital status at diagnosis', 'Lung - Pleural/Elastic Layer Invasion (PL) by H and E or Elastic Stain', 'Lung - Separate Tumor Nodules - Ipsilateral Lung', 'Lung - Surgery to Primary Site (1988-2015)', 'Lung - Surgery to Other Regional/Distant Sites (1998+)']
print('Columns to apply One-Hot Encoding:\n {}'.format(cols))
for column in cols:
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column, prefix_sep='=', dummy_na=False, dtype=np.uint8)], axis=1).drop([column], axis=1)

print(df.info())
print(df.head())

df.to_csv('./data/train_LCCS_onehot.csv', encoding='utf-8', index=False)

