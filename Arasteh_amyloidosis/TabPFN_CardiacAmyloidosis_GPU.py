exec(open("../TabPFN_CommonScript-Top.py").read())

# Load Data
df = pd.read_csv("../training_input.csv")

# Define target and drop col list
VOI = 'cohort_flag'
droplist = [VOI,
            "patient_id", 
            "cohort_type"]

# Ensure numeric columns remain numeric after fillna
df2 = clean_missing_values(df)

old = '20250101-0000-00'
RUN_TABPFN_HPO = True
KeepSplit = False

exec(open("../TabPFN_CommonScript-Bottom_GPU.py").read())