# Import 
print("importing libraries")

# === Config flags ===
# Additional Boosting Models (requires installation)
# try:
#     from xgboost import XGBClassifier
# except ImportError:
#     print("XGBoost is not installed. Install it using: pip install xgboost")

# try:
#     from lightgbm import LGBMClassifier
# except ImportError:
#     print("LightGBM is not installed. Install it using: pip install lightgbm")

# try:
#     from catboost import CatBoostClassifier
# except ImportError:
#     print("CatBoost is not installed. Install it using: pip install catboost")

from tabpfn_extensions.hpo import TunedTabPFNClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import mean_squared_error
from tabpfn_extensions import interpretability
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_diabetes
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.utils import all_estimators
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score 
from sklearn.metrics import r2_score
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from sklearn.utils import resample
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from datetime import datetime
from inspect import signature
from sklearn.svm import SVC
from pandasql import sqldf
from pickle import FALSE
from scipy import stats
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import sqlite3
import psutil
import pynvml
import torch
import json
import glob
import time
import sys
import os
import re

try:
    import pynvml
    pynvml.nvmlInit()
except Exception as e:
    print(f"⚠️ NVML not available: {e}")

print("setting date toime")
now = datetime.now().strftime("%Y%m%d-%H%M-%S")
warnings.filterwarnings("ignore", category=FutureWarning)

print("making functions")
# Functions
def clean_missing_values(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(-1).astype(float)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].fillna(pd.Timestamp("1900-01-01"))
            df[col] = df[col].dt.strftime('%Y%m%d').astype(float)
        else:
            df[col] = df[col].fillna("Missing")
    return df

def auto_encode_non_numeric(df):
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object' or df_cleaned[col].dtype.name == 'category':
            df_cleaned[col] = df_cleaned[col].astype('category').cat.codes
    return df_cleaned

print("messing with CUDA")

if torch.cuda.is_available():
    try:
        # torch.cuda.set_device(1) 
        print(f"Current CUDA Device Index: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    except Exception as e:
        print(f"Warning: Unable to set CUDA device 0 — {e}")
else:
    print("CUDA not available.")

print("Making some more functions")

def smart_clean(df, verbose=True):

    df_clean = df.copy()
    dropped_cols = []
    converted_dates = []

    # Drop columns with >95% missing or identical
    for col in df_clean.columns:
        null_frac = df_clean[col].isna().mean()
        if null_frac > 0.95:
            dropped_cols.append(col)
            df_clean.drop(columns=col, inplace=True)
            continue
        if df_clean[col].nunique(dropna=True) <= 1:
            dropped_cols.append(col)
            df_clean.drop(columns=col, inplace=True)

    # Drop high-cardinality object columns
    for col in df_clean.select_dtypes(include=['object', 'category']).columns:
        if df_clean[col].nunique() > 100:
            dropped_cols.append(col)
            df_clean.drop(columns=col, inplace=True)

    # Convert date-like strings to numeric
    origin = pd.Timestamp("1900-01-01")
    for col in df_clean.select_dtypes(include=['object']).columns:
        try:
            parsed = pd.to_datetime(df_clean[col], format="%Y-%m-%d", errors='coerce')
            if parsed.notna().sum() > 100:
                df_clean[col] = (parsed - origin).dt.days
                converted_dates.append(col)
        except Exception:
            continue

    if verbose:
        print(f"Cleaned shape: {df_clean.shape}")
        print("\nDropped columns:")
        for col in dropped_cols:
            print(f" - {col}")
        print("\nConverted date columns:")
        for col in converted_dates:
            print(f" - {col}")

    return df_clean, dropped_cols

def clean_df_train(df_train):
    # Replace stealthy nulls
    stealth_nulls = [
        "Blank(s)", "Unknown", "Not applicable", "Recode not available",
        "None; diagnosed at autopsy", "No/Unknown", "999", "888", "Missing"
    ]
    df_train.replace(stealth_nulls, np.nan, inplace=True)

    # Drop constant columns
    constant_cols = [col for col in df_train.columns if df_train[col].nunique(dropna=True) == 1]
    df_train.drop(columns=constant_cols, inplace=True)

    # Encode categorical columns
    label_encoders = {}
    for col in df_train.select_dtypes(include='object').columns:
        le = LabelEncoder()
        try:
            df_train[col] = le.fit_transform(df_train[col].astype(str))
            label_encoders[col] = le
        except Exception as e:
            print(f"Could not encode {col}: {e}")


    df_train.fillna(-9, inplace=True)

    return df_train

def get_completed_bootstraps(csv_file, n_bootstraps):
    if not os.path.exists(csv_file):
        return set()
    try:
        df = pd.read_csv(csv_file)
        done = set(df['Bootstrap'])
        return done
    except Exception as e:
        print(f"Could not read {csv_file}: {e}")
        return set()