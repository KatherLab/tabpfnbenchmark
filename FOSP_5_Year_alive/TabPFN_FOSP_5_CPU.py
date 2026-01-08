exec(open("../TabPFN_CommonScript.py").read())

# Read in df [Aquire access to FOSP, read in data here]
df = pd.read_csv("")

# Filter down to CRC
df2 = df[df["TOPO"].str.startswith(("C18", "C19", "C20"), na=False)].copy()

# Overall death: if ULTINFO is 3 or 4
df2["overall_death"] = df2["ULTINFO"].apply(lambda x: 1 if str(x) in ["3", "4"] else 0)

# Cancer-specific death: only if ULTINFO is 3
df2["cancer_death"] = df2["ULTINFO"].apply(lambda x: 1 if str(x) == "3" else 0)

# Ensure both columns are datetime
df2["DTDIAG"] = pd.to_datetime(df2["DTDIAG"], errors="coerce")
df2["DTULTINFO"] = pd.to_datetime(df2["DTULTINFO"], errors="coerce")

# Calculate days between diagnosis and last info
df2["days_survived"] = (df2["DTULTINFO"] - df2["DTDIAG"]).dt.days

df2["alive_year1"] = np.where(
    df2["days_survived"] >= 365, 1,
        np.where(df2["ULTINFO"].astype(str).isin(["3", "4"]), 0, np.nan))

df2["alive_year3"] = np.where(
    df2["days_survived"] >= 1095, 1,
        np.where(df2["ULTINFO"].astype(str).isin(["3", "4"]), 0, np.nan))

df2["alive_year5"] = np.where(
    df2["days_survived"] >= 1825, 1,
        np.where(df2["ULTINFO"].astype(str).isin(["3", "4"]), 0, np.nan))

VOI = 'alive_year5'

df2 = df2.dropna(subset=[VOI])

# Ensure numeric columns remain numeric after fillna
non_string_columns = [col for col in df2.columns if df2[col].dtype != 'object']
for col in df2.columns:
    if pd.api.types.is_numeric_dtype(df2[col]):
        df2[col] = df2[col].fillna(-1).astype(float)
    elif pd.api.types.is_datetime64_any_dtype(df2[col]):
        df2[col] = df2[col].fillna(pd.Timestamp("1900-01-01"))
        df2[col] = df2[col].dt.strftime('%Y%m%d').astype(float)
    else:
        df2[col] = df2[col].fillna("Missing")

df2["cancer_death"] = df2["ULTINFO"].apply(lambda x: 1 if str(x) == "3" else 0)
df2["days_survived"] = (df2["DTULTINFO"] - df2["DTDIAG"])

droplist = [VOI,
            "days_survived", 
            "DTULTINFO", 
            "ULTINFO",
            "overall_death",
            "cancer_death",
            "alive_year3",
            "alive_year5"
            ]
old = '20250101-0000-00'
RUN_TABPFN_HPO = False

df2, dropped = smart_clean(df2)
print("Dropped columns:", dropped)
droplist = [col for col in droplist if col not in dropped]

KeepSplit = False

exec(open("../TabPFN_CommonScript-Bottom_CPU.py").read())