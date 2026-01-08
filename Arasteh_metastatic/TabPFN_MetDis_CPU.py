exec(open("../TabPFN_CommonScript.py").read())

# Read in df [Download from Arasteh et al]

df = pd.read_csv("MedDis_DataZonodo.csv")
df = df.dropna(subset=['Patient ID'])
df['Metastatic'] = np.where(df['Metastatic YES/NO'] == 'YES', 1, 0)

# Set Target
VOI = 'Metastatic'
droplist = [VOI,
            "Predictions for metastatic disease", 
            "Metastatic YES/NO", "Patient ID"]

df2 = clean_missing_values(df)

old = '20250101-0000-00'
RUN_TABPFN_HPO = False 
KeepSplit = Falses

exec(open("../TabPFN_CommonScript-Bottom_CPU.py").read())