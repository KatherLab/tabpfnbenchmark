exec(open("../TabPFN_CommonScript.py").read())

# Read in df [Aquire OST data from SEER]
df_path = ""
df2 = pd.read_csv(df_path)
df2 = df2.drop(columns=[col for col in df2.columns if (df2[col] == 'Blank(s)').all()])

df2["5_Year_Survival"] = np.where(df2["Survival months"] >= 60, 1,
        np.where(df2["Vital status recode (study cutoff used)"].astype(str).isin(["Dead"]), 0, np.nan))

VOI = '5_Year_Survival'

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
    "Survival months",
    "Survival months flag",
    "Vital status recode (study cutoff used)",
    "COD to site recode ICD-O-3 2023 Revision Expanded (1999+)",
    "COD to site recode ICD-O-3 2023 Revision",
    "SEER cause-specific death classification",
    "COD to site recode",
    "COD to site rec KM",
    "SEER other cause of death classification"    
]


old = '20250101-0000-00'
KeepSplit = False
RUN_TABPFN_HPO = False  
clean_df_train(df2)
filtered_droplist = [col for col in droplist if col in df2.columns]
droplist = filtered_droplist

exec(open("../TabPFN_CommonScript-Bottom_GPU.py").read())