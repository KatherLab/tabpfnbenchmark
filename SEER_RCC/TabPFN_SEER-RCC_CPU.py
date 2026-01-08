exec(open("../TabPFN_CommonScript.py").read())

# Read in df [Aquire RCC data from SEER]
dfath = "RCC_test2_LAS_20250520.csv"
df2 = pd.read_csv(dfath)
df2 = df2.drop(columns=[col for col in df2.columns if (df2[col] == 'Blank(s)').all()])

df2['Survival months'] = pd.to_numeric(df2['Survival months'], errors='coerce')
df2["Survival months bi"] = np.where(df2["Survival months"] >= 60, 1,
        np.where(df2["Vital status recode (study cutoff used)"].astype(str).isin(["Dead"]), 0, np.nan))

VOI = 'Survival months bi'

for df in [df2]:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

for col in df2.columns:
    if pd.api.types.is_numeric_dtype(df2[col]):
        df2[col] = df2[col].fillna(0).astype(float)
    elif pd.api.types.is_datetime64_any_dtype(df2[col]):
        df2[col] = df2[col].fillna(pd.Timestamp("1900-01-01"))
        df2[col] = df2[col].dt.strftime('%Y%m%d').astype(float)
    else:
        df2[col] = df2[col].fillna("Missing")

# Drop List
droplist = [
    VOI,
    "SEER other cause of death classification",
    "SEER cause-specific death classification",
    "COD to site recode ICD-O-3 2023 Revision Expanded (1999+)",
    "Survival months flag",
    "Survival months",
    "Vital status recode (study cutoff used)",
    "COD to site recode",
    "COD to site recode ICD-O-3 2023 Revision",
    "COD to site rec KM"
]

old = '20250717-0508-50'
RUN_TABPFN_HPO = False  
KeepSplit = False  
clean_df_train(df2)
filtered_droplist = [col for col in droplist if col in df2.columns]
droplist = filtered_droplist

exec(open("../TabPFN_CommonScript-Bottom_CPU.py").read())