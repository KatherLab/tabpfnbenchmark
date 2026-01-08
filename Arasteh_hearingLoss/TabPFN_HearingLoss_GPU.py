exec(open("../TabPFN_CommonScript.py").read())

# Read in df [Download from Arasteh et al]
df_train_path = "DiscoverySet.xlsx"
df_test_path = "ValidationSet.xlsx"

df_test = pd.read_excel(df_test_path)
df_train = pd.read_excel(df_train_path)

VOI = 'Diagnoses'
target_column = VOI

df_test = clean_missing_values(df_test)
df_train = clean_missing_values(df_train)

target_column = VOI

droplist = [VOI]

#train_df
X_train = df_train.drop(columns=droplist)
y_train = df_train[[VOI]]

#test_df
X_test = df_test.drop(columns=droplist)
y_test = df_test[[VOI]]

# Fix mixed-type column
for df in [X_train, X_test]:
    if "c.235delC" in df.columns:
        df["c.235delC"] = pd.to_numeric(df["c.235delC"], errors="coerce").fillna(0.0).astype(float)


# %%
y_test = y_test[[VOI]].astype(int)
y_train = y_train[[VOI]].astype(int)

print(y_train[VOI].value_counts())
print(y_test[VOI].value_counts())

for df in [X_train, X_test]:
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


# Keep target numeric as before
y_test = y_test.apply(pd.to_numeric, errors='coerce').fillna(0)
y_train = y_train.apply(pd.to_numeric, errors='coerce').fillna(0)

# adaptor
df_train['c.235delC'] = pd.to_numeric(df_train['c.235delC'], errors='coerce')
df_train = df_train.loc[:, df_train.nunique() > 1]
sparse_threshold = 0.005  # i.e., <0.5% ones
to_drop = [col for col in df_train.columns if (df_train[col].mean() < sparse_threshold) and col != 'Diagnoses']
df_train.drop(columns=to_drop, inplace=True)
df_train.fillna(0, inplace=True)  # or use sklearn's SimpleImputer


old = '20250101-0000-00'
RUN_TABPFN_HPO = False  
KeepSplit = True

exec(open("../TabPFN_CommonScript-Bottom_GPU.py").read())