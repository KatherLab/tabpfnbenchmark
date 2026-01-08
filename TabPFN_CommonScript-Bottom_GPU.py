# Configuration flags
# Store results       # Default values
random_state = 91311  # 91311
n_bootstraps = 5      # 500
n_trials = 50         # 50
PCA_size = 2000       # 2000 Max
test_size = 0.3       # 30%
row_limit = 50000     # 50000

if KeepSplit:
    print("Using pre split data")
else:

    # Use full dataset for all models
    X = df2.drop(columns=droplist)
    y = df2[VOI]
    X.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in X.columns]

    # Split into train and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size = test_size,         # 30% test data
        random_state = random_state,   # for reproducibility
        stratify = y                   # optional, keeps class balance
    )

    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)


# --- Limit to first 50000 rows if needed ---
if X_train.shape[0] >= row_limit:
    print(f"X_train has {X_train.shape[0]} rows, randomly sampling {row_limit} rows...")
    sample_idx = X_train.sample(n=row_limit, random_state=random_state).index
    X_train = X_train.loc[sample_idx]
    y_train = y_train.loc[sample_idx]
else:
    print(f"X_train has {X_train.shape[0]} rows, no row limit needed.")

if X_test.shape[0] >= row_limit:
    print(f"X_test has {X_test.shape[0]} rows, randomly sampling {row_limit} rows...")
    sample_idx = X_test.sample(n=row_limit, random_state=random_state).index
    X_test = X_test.loc[sample_idx]
    y_test = y_test.loc[sample_idx]
else:
    print(f"X_test has {X_test.shape[0]} rows, no row limit needed.")

# One Hot Encode Cat Data

# Identify categorical and numerical features (add debug prints)
categorical_features_train = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features_train = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features_test = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features_test = X_test.select_dtypes(include=['number']).columns.tolist()

print("Categorical features (x_train):", categorical_features_train)
print("Numerical features (x_train):", numerical_features_train)
print("Categorical features (x_test):", categorical_features_test)
print("Numerical features (x_test):", numerical_features_test)

if len(categorical_features_train) > 0:
    print("Encoding categorical features")
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_train = enc.fit_transform(X_train[categorical_features_train])
    X_num_train = X_train[numerical_features_train].values if numerical_features_train else None
    if X_num_train is not None:
        X_train = np.hstack([X_num_train, X_cat_train])
    else:
        X_train = X_cat_train
    X_train = pd.DataFrame(X_train, index=y_train.index)
    print("Shape after encoding:", X_train.shape)

if len(categorical_features_test) > 0:
    print("Encoding categorical features")
    # enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_test = enc.transform(X_test[categorical_features_test])
    X_num_test = X_test[numerical_features_test].values if numerical_features_test else None
    if X_num_test is not None:
        X_test = np.hstack([X_num_test, X_cat_test])
    else:
        X_test = X_cat_test
    X_test = pd.DataFrame(X_test, index=y_test.index)
    print("Shape after encoding:", X_test.shape)

# # --- Apply PCA if more than {PCA_size} columns ---
USED_PCA = True

#  Add PCA if encoding pushes us over the feature limit
if X_train.shape[1] > PCA_size:
    print("Starting PCA...")
    pca = PCA(n_components=PCA_size, random_state=random_state)
    X_pca_encoded_train = pca.fit_transform(X_train)
    print(f"Shape after encoding is {X_train.shape[1]} features, applying post-encoding PCA to reduce to {PCA_size}...")
    X_train = pd.DataFrame(X_pca_encoded_train, index=y_train.index, columns=[f'PC{i+1}' for i in range(PCA_size)])
    USED_PCA = True
else:
    print(f"Shape after encoding is {X_train.shape[1]} features, no additional PCA needed.")
    USED_PCA = False

#  Apply train PCA to test set if it was used
if USED_PCA:
    print("Applying training PCA to test set...")
    X_pca_encoded_test = pca.transform(X_test)
    X_test = pd.DataFrame(X_pca_encoded_test, index=y_test.index, columns=[f'PC{i+1}' for i in range(PCA_size)])
    print(f"Test set transformed using training PCA. New shape: {X_test.shape}")
else:
    print(f"Shape after encoding is {X_test.shape[1]} features, no additional PCA needed.")

# Check for empty features before proceeding
if (X_train.shape[1] == 0) or (X_test.shape[1] == 0):
    raise ValueError("No features available after preprocessing! Check your droplist and input data.")

# At this point, X is fully numeric and ready for modeling
# Check if existing results file exists
now = datetime.now().strftime("%Y%m%d-%H%M-%S")
print(now)

# --- TabPFN HPO step (toggleable) ---
# Add HPO checkpointing
hpo_params_file = f"{VOI}_TabPFN_HPO_best_params.json"
hpo_tuned_params = {}

if RUN_TABPFN_HPO:
    if os.path.exists(hpo_params_file):
        print(f"Found cached HPO params at {hpo_params_file}, loading...")
        with open(hpo_params_file, 'r') as f:
            hpo_tuned_params = json.load(f)
        print(f"Loaded HPO params: {hpo_tuned_params}")
    else:
        print("\nRunning TabPFN Hyperparameter Optimization (HPO) before main ML loop...")
        # Run HPO
        hpo_clf = TunedTabPFNClassifier(device='cuda', n_trials=n_trials, metric='f1', random_state=random_state)
        hpo_clf.fit(X_train, y_train.values.ravel())
        hpo_best_model = hpo_clf.best_model_
        valid_keys = signature(TabPFNClassifier.__init__).parameters.keys()
        hpo_tuned_params = {k: v for k, v in hpo_best_model.get_params().items() if k in valid_keys}
        print(f"TabPFN HPO completed. Best parameters: {hpo_tuned_params}")
        # Save to file
        with open(hpo_params_file, 'w') as f:
            json.dump(hpo_tuned_params, f)
        print(f"HPO params saved to {hpo_params_file}")
else:
    print("Skipping TabPFN Hyperparameter Optimization (HPO) step.")
    hpo_tuned_params = {}

# List of classifiers to test

tabpfn_model_path = "../tabpfn-v2.5-classifier-v2.5_default.ckpt"

#  Sanity check
if not os.path.exists(tabpfn_model_path):
    raise FileNotFoundError(f"TabPFN model not found at: {tabpfn_model_path}")

#  Remove any unsupported HPO params
if "features_per_group" in hpo_tuned_params:
    print("Removing invalid HPO key: features_per_group")
    del hpo_tuned_params["features_per_group"]

#  Define models with safe config
models = {
    "TabPFN": TabPFNClassifier(model_path=tabpfn_model_path),
    "TabPFN-HPO": TabPFNClassifier(model_path=tabpfn_model_path, **hpo_tuned_params)
}

existing_results_file = f"{VOI}_Youden_model_metrics_LAS_{old}.csv"
print(f"Checking for {existing_results_file}")

# Check for any existing results file with the same VOI
existing_files = glob.glob(f"{VOI}_Youden_model_metrics_LAS_*.csv")

if existing_files:
    print(f"Found existing results files: {existing_files}")
    use_existing = input("Do you want to use existing results? (y/n): ").lower().strip()
    
    if use_existing == 'y':
        # Load the most recent file
        most_recent_file = max(existing_files, key=os.path.getctime)
        print(f"Loading existing results from: {most_recent_file}")
        
        # Load the summary results (this is all we need!)
        results_df = pd.read_csv(most_recent_file)
        summary_rows = results_df.to_dict('records')
        
        print("Successfully loaded existing results. Skipping ML loop.")
        run_ml_loop = False
    else:
        print("Will run new ML evaluation loop.")
        summary_rows = []
        run_ml_loop = True
else:
    print("No existing results found. Will run new ML evaluation loop.")
    summary_rows = []
    run_ml_loop = True

roc_buffers = []  # Store last 5 ROC curves (fpr, tpr, auc)

# Run ML loop only if needed
if run_ml_loop:
    print("\nStarting ML evaluation loop...")
    
    model_runtime_loops = {}
    for model_name, model in models.items():
        # --- Resume logic: find most recent bootstrap_metrics file for this VOI/model ---
        pattern = f"{VOI}_{model_name}_bootstrap_metrics_LAS_*.csv"
        existing_bootstrap_files = glob.glob(pattern)
        if existing_bootstrap_files:
            # Use the most recent file for resuming/appending
            bootstrap_metrics_file = max(existing_bootstrap_files, key=os.path.getctime)
            print(f"Resuming from existing file: {bootstrap_metrics_file}")
        else:
            # No file exists, create a new one
            now = datetime.now().strftime("%Y%m%d-%H%M-%S")
            bootstrap_metrics_file = f"{VOI}_{model_name}_bootstrap_metrics_LAS_{now}.csv"
            print(f"Creating new file: {bootstrap_metrics_file}")
        # --- End resume logic ---
        print(f"Checking for {bootstrap_metrics_file}")
        completed = get_completed_bootstraps(bootstrap_metrics_file, n_bootstraps)
        print(f"{len(completed)} bootstraps already completed for {model_name}.")
        all_bootstrap_metrics = []
        loop_runtimes = []

        try:
            for b in range(n_bootstraps):
                # Skip if this bootstrap already done
                if (b + 1) in completed:
                    continue

                # Bootstrap sample
                X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=random_state + b, stratify=y_train)
                bootstrap_metrics = []

                #  minimal change: replace fold CV with one train/test split
                loop_start_time = time.time()
                print(f"Bootstrap {b+1}/{n_bootstraps}")

                # No further preprocessing needed, X_boot and X_test_fold are ready
                print(f"Model input shape: {X_boot.shape[0]} rows × {X_boot.shape[1]} features{' (PCA applied)' if USED_PCA else ''}")

                # Train on bootstrapped training data
                model.fit(
                    X_boot,
                    y_boot.values.ravel() if hasattr(y_boot, 'values') else y_boot
                )

                # Test on the fixed hold-out set
                y_pred_default = model.predict(X_test)
                y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                # Calculate metrics with default threshold
                accuracy_default = accuracy_score(y_test, y_pred_default)
                balanced_acc_default = balanced_accuracy_score(y_test, y_pred_default)
                f1_default = f1_score(y_test, y_pred_default, average='weighted')
                print(f"[INFO] Metrics for {model_name}: Accuracy={accuracy_default:.4f}, Balanced Acc={balanced_acc_default:.4f}, F1={f1_default:.4f}")

                bootstrap_metrics = []
                if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                    # --- Compute ROC and Youden threshold ---
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
                    youden_threshold = thresholds[np.argmax(tpr - fpr)]
                    y_pred_adjusted = (y_prob[:, 1] >= youden_threshold).astype(int)

                    # --- Compute adjusted metrics ---
                    adjusted_accuracy = accuracy_score(y_test, y_pred_adjusted)
                    adjusted_balanced_acc = balanced_accuracy_score(y_test, y_pred_adjusted)
                    adjusted_f1 = f1_score(y_test, y_pred_adjusted, average='weighted')
                    adjusted_auc = roc_auc_score(y_test, y_prob[:, 1])
                    precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
                    auc_pr = auc(recall, precision)

                    # --- Store last 5 ROC curves for plotting ---
                    roc_buffers.append((fpr, tpr, adjusted_auc))
                    if len(roc_buffers) > 5:
                        roc_buffers.pop(0)

                    # --- Logging ---
                    print(f"[INFO] Youden threshold for {model_name}: {youden_threshold:.4f}")
                    print(f"[INFO] Adjusted metrics: Accuracy={adjusted_accuracy:.4f}, "
                        f"Balanced Acc={adjusted_balanced_acc:.4f}, "
                        f"F1={adjusted_f1:.4f}, AUC={adjusted_auc:.4f}, AUC-PR={auc_pr:.4f}")

                    # --- Save metrics to file ---
                    bootstrap_metrics.append({
                        "Model": model_name,
                        "Bootstrap": b + 1,
                        "AUC": adjusted_auc,
                        "AUC-PR": auc_pr,
                        "Accuracy": accuracy_default,
                        "Accuracy(Youden)": adjusted_accuracy,
                        "Balanced-Accuracy": balanced_acc_default,
                        "Balanced-Accuracy(Youden)": adjusted_balanced_acc,
                        "F1-score": f1_default,
                        "F1-score(Youden)": adjusted_f1,
                        "Threshold": youden_threshold
                    })
                    print(bootstrap_metrics)

                else:
                    print(f"{model_name} is NOT YOUDEN COMPATIBLE (no probability estimates)")

                # Runtime bookkeeping
                loop_end_time = time.time()
                bootstrap_runtime = loop_end_time - loop_start_time
                if bootstrap_metrics:
                    for metric_dict in bootstrap_metrics:
                        metric_dict["Bootstrap Runtime (s)"] = bootstrap_runtime

                    # Save immediately (append or create header)
                    pd.DataFrame(bootstrap_metrics).to_csv(
                        bootstrap_metrics_file,
                        mode='a',
                        header=not os.path.exists(bootstrap_metrics_file) or os.path.getsize(bootstrap_metrics_file) == 0,
                        index=False
                    )

                    all_bootstrap_metrics.extend(bootstrap_metrics)
                loop_runtimes.append(bootstrap_runtime)

            # === Save last 5 AUC curves as SVG ===
            if roc_buffers:
                plt.figure(figsize=(6, 5))
                for i, (fpr, tpr, auc_val) in enumerate(roc_buffers, 1):
                    plt.plot(fpr, tpr, label=f'Run -{5 - i + 1}: AUC={auc_val:.3f}')
                mean_auc = np.mean([x[2] for x in roc_buffers])
                std_auc = np.std([x[2] for x in roc_buffers])
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                plt.title(f"{VOI} – {model_name}\nAUC: {mean_auc:.3f} ± {std_auc:.3f}")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                plt.tight_layout()
                fname = f"AUC_{VOI}_{model_name}_last5.svg"
                plt.savefig(fname, format='svg', bbox_inches='tight')
                plt.close()
                print(f"Saved last 5 AUC curves for {model_name}: {fname}")

            model_runtime_loops[model_name] = loop_runtimes            

            # Aggregate per-model summary if we collected any metrics
            if all_bootstrap_metrics:
                df_all = pd.DataFrame(all_bootstrap_metrics)
                df_all.to_csv(f"{VOI}_{model_name}_bootstrap_metrics_LAS_{now}.csv", index=False)

                # Drop identifiers; "Fold" may not exist — drop if present
                cols_to_drop = [c for c in ["Model", "Bootstrap", "Fold"] if c in df_all.columns]
                summary = df_all.drop(columns=cols_to_drop).agg(["mean", "std"])

                n = len(df_all)
                final_row = {"Model": model_name}
                for metric in summary.columns:
                    mean_val = summary.loc["mean", metric]
                    std_val = summary.loc["std", metric]
                    ci_low, ci_high = stats.t.interval(
                        confidence=0.95, df=n - 1, loc=mean_val, scale=std_val / np.sqrt(n)
                    )
                    final_row[f"{metric} Mean"] = mean_val
                    final_row[f"{metric} Std"] = std_val
                    final_row[f"{metric} CI Low"] = ci_low
                    final_row[f"{metric} CI High"] = ci_high

                # Add mean runtime per loop and std error
                if loop_runtimes:
                    mean_runtime = np.mean(loop_runtimes)
                    std_runtime = np.std(loop_runtimes, ddof=1) if len(loop_runtimes) > 1 else 0.0
                    stderr_runtime = std_runtime / np.sqrt(len(loop_runtimes)) if len(loop_runtimes) > 1 else 0.0
                else:
                    mean_runtime = 0.0
                    std_runtime = 0.0
                    stderr_runtime = 0.0

                final_row["Avg Runtime per Loop (s)"] = mean_runtime
                final_row["Runtime Std (s)"] = std_runtime
                final_row["Runtime StdErr (s)"] = stderr_runtime
                summary_rows.append(final_row)

        except Exception as e:
            print(f"Error with {model_name}: {e}")

    results_df = pd.DataFrame(summary_rows)
    results_df.to_csv(f"{VOI}_Youden_model_metrics_{now}.csv", index=False)
    results_df.rename(columns={"index": "Model"}, inplace=True)
    print(f"\nFinal Model Metrics saved as {VOI}_Youden_model_metrics_{now}.csv.")

    # === Barplot of Model AUC Scores ===
    # Clean and filter results
    results_df3 = results_df.dropna(subset=["AUC Mean", "AUC Std"])
    results_df3 = results_df3[~results_df3['Model'].isin(['RidgeClassifier', 'TabICL'])]
    results_df3 = results_df3.drop_duplicates(subset=["Model"])

    # Prepare data
    plot_df = results_df3.sort_values(by='AUC Mean', ascending=False)[['Model', 'AUC Mean', 'AUC Std']]
    plot_df.to_csv(f"{VOI}_ALL_PLUS_5Fold_model_metrics_LAS_{now}.csv", index=False)

    # Plot
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        x='AUC Mean', y='Model', data=plot_df,
        palette='viridis', xerr=plot_df['AUC Std']
    )

    # Annotate with text labels to the right of whiskers
    for bar, mean, std in zip(ax.patches, plot_df['AUC Mean'], plot_df['AUC Std']):
        whisker_tip = mean + std
        ax.text(whisker_tip + 0.01, bar.get_y() + bar.get_height()/2,
                f"{mean:.4f}", va='center', ha='left', fontsize=10)

    # Add reference lines
    for x in np.arange(0.1, 1.0, 0.1):
        plt.axvline(x=x, color='gray', linestyle='dotted', linewidth=0.75)

    # Label map
    label_map = {
        'cancer_death': 'Cancer_Death',
        'overall_death': 'Overall_Death',
        'alive_year1': 'Alive_Years_1',
        'alive_year3': 'Alive_Years_3',
        'alive_year5': 'Alive_Years_5',
        'cohort_flag': 'Cardiac Amyloidosis',
        'GroundTruth_bi': 'Esophageal Cancer',
        'Metastatic': 'Metastatic Disease'
    }
    Label = label_map.get(VOI, VOI)

    # Finalize
    plt.xlim(0, 1.05)
    plt.title(f'Model AUC Scores: {Label} (n_bootstraps={n_bootstraps})', fontsize=16)
    plt.xlabel('AUC (Mean ± Std)', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{VOI}_BarPlot_LAS_{now}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # === Barplot of Model Runtimes (per loop) ===
    # Prepare runtime data
    runtime_df = results_df.dropna(subset=["Avg Runtime per Loop (s)"])
    runtime_df = runtime_df[~runtime_df['Model'].isin(['RidgeClassifier', 'TabICL'])]
    runtime_df = runtime_df.drop_duplicates(subset=["Model"])
    runtime_df = runtime_df.sort_values(by="Avg Runtime per Loop (s)", ascending=False)

    plt.figure(figsize=(10, 8))
    ax = sns.barplot(
        x='Avg Runtime per Loop (s)', y='Model', data=runtime_df,
        palette='mako', xerr=runtime_df['Runtime StdErr (s)']
    )
    for bar, mean_runtime, stderr in zip(ax.patches, runtime_df['Avg Runtime per Loop (s)'], runtime_df['Runtime StdErr (s)']):
        ax.text(mean_runtime + stderr + 0.5, bar.get_y() + bar.get_height()/2,
                f"{mean_runtime:.2f}s", va='center', ha='left', fontsize=10)
    plt.title(f'Model Runtime per Loop (n_bootstraps={n_bootstraps})', fontsize=16)
    plt.xlabel('Average Runtime per Loop (seconds)', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{VOI}_RuntimePerLoop_BarPlot_LAS_{now}.png', dpi=300, bbox_inches='tight')
    plt.show()

else:
    print("Using existing results. ML loop skipped.")