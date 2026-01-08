exec(open("../TabPFN_CommonScript.py").read())

# Read in df [Aquire HCC data through SEER]
df_train_path = ""
df2 = pd.read_csv(df_train_path)
df2 = df2.drop(columns=[col for col in df2.columns if (df2[col] == 'Blank(s)').all()])

df2["LungMets"] = np.where(df2['SEER Combined Mets at DX-lung (2010+)'] == 'Yes', 1, 0)

VOI = 'LungMets'

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
    "SEER Combined Mets at DX-bone (2010+)",
    "SEER Combined Mets at DX-lung (2010+)", 
    VOI,
    "CS mets at dx (2004-2015)",
    "EOD Mets (2018+)",
    "Mets at DX-Other (2016+)",
    "Mets at DX-Distant LN (2016+)",
    "CS Mets Eval (2004-2015)",
    "Combined Summary Stage (2004+)",
    "Derived EOD 2018 M (2018+)",
    "SEER Combined Summary Stage 2000 (2004-2017)"
    ,"Histologic Type ICD-O-3"
    ,"Derived EOD 2018 Stage Group (2018+)"
    ,"Time from diagnosis to treatment in days recode"
    ,"SEER Combined Mets at DX-brain (2010+)"
    ,"SEER Combined Mets at DX-liver (2010+)"
    ,"Year of diagnosis"
    ,"Summary stage 2000 (1998-2017)"
    ,"SEER historic stage A (1973-2015)"
]

clean_df_train(df2)

old = '20250101-0000-00'
RUN_TABPFN_HPO = False 
KeepSplit = False

exec(open("../TabPFN_CommonScript-Bottom_CPU.py").read())