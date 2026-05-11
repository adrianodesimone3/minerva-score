"""
Data Preprocessing Pipeline for Biliary Pancreatitis Study
===========================================================

This script performs:
1. Data loading and standardization of column names
2. Alignment of variables between prospective and retrospective datasets
3. Train/Validation/Test split (70/15/15) with stratification
4. MICE imputation for missing values
5. Normalization of continuous variables
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pickle
import os

print("=" * 80)
print("BILIARY PANCREATITIS DATA PREPROCESSING PIPELINE")
print("=" * 80)


# =============================================================================
# COLUMN MAPPING AND STANDARDIZATION
# =============================================================================

# Map retrospective columns to prospective naming convention
COLUMN_MAPPING = {
    'Patient age': 'age',
    'Sex': 'sex',
    'Previous episodes of biliary pancreatitis': 'previous_episodes',
    'Admitting speciality': 'admitting_specialty',
    'Body Mass Index - BMI (Kg/m2)': 'bmi',
    'Clinical history of diabetes': 'diabetes',
    'Clinical history of chronic pulmonary disease (other than Covid-19 pneumonia)': 'chronic_pulmonary_disease',
    'Clinical history of hypertension': 'hypertension',
    'Clinical history of atrial fibrillation': 'atrial_fibrillation',
    'Clinical history of ischemic heart disease': 'ischemic_heart_disease',
    'Clinical history of chronic kidney disease ': 'chronic_kidney_disease',  # Note trailing space
    'Clinical history of diseases of the hematopoietic system': 'hematopoietic_disease',
    'Patient on immu0suppressive medications on hospital admission': 'immunosuppressive_medications',  # Fix typo
    'WBC on admission (cells/mm3)': 'wbc',
    'Neutrophils on admission (cells/mm3)': 'neutrophils',
    'Platelets on admission (mcL)': 'platelets',
    'INR - International Normalized Ratio on admission': 'inr',
    'C-Reactive Protein on admission (mg/L)': 'crp',
    'Aspartate aminotransferase-AST on admission (U/L)': 'ast',
    'Alanine aminotransferase-ALT on admission (U/L)': 'alt',
    'Total bilirubin on admission (mg/dL)': 'total_bilirubin',
    'Conjugated bilirubin on admission (mg/dL)': 'conjugated_bilirubin',
    'Gamma-glutamyl transpeptidase-GGT on admission (U/L)': 'ggt',
    'Serum amylase on admission (U/L)': 'serum_amylase',
    'Serum lipase on admission (U/L)': 'serum_lipase',
    'Lactate DeHydrogenase-LDH on admission (U/L)': 'ldh',
    'Choledocholithiasis': 'choledocholithiasis',
    'Cholangitis': 'cholangitis',
    'ERCP': 'ercp',
    '30-day Hospital readmission': 'target'
}

# Map prospective columns to standardized names
PROSPECTIVE_MAPPING = {
    'eta_del_paziente': 'age',
    'sesso_del_paziente': 'sex',
    'precedenti_episodi_di_panc': 'previous_episodes',
    'reparto_di_ricovero': 'admitting_specialty',
    'indice_di_massa_corporea_b': 'bmi',
    'storia_clinica_di_diabete': 'diabetes',
    'storia_clinica_di_malattia': 'chronic_pulmonary_disease',
    'storia_clinica_di_ipertens': 'hypertension',
    'storia_clinica_di_fibrilla': 'atrial_fibrillation',
    'storia_clinica_di_cardiopa': 'ischemic_heart_disease',
    'storia_clinica_di_malattia_ren': 'chronic_kidney_disease',
    'storia_clinica_di_malattia_emato': 'hematopoietic_disease',
    'trattamento_con_farmaci_im': 'immunosuppressive_medications',
    'globuli_bianchi_wbc_cellul': 'wbc',
    'neutrofili_cellule_mm3': 'neutrophils',
    'piastrine_plt_mm3': 'platelets',
    'inr_rapporto_internazional': 'inr',
    'proteina_c_reattiva_pcr_mg': 'crp',
    'aspartato_aminotransferasi': 'ast',
    'alanina_aminotransferasi_a': 'alt',
    'bilirubina_totale_mg_dl': 'total_bilirubin',
    'bilirubina_coniugata_diret': 'conjugated_bilirubin',
    'gamma_glutamil_transpeptid': 'ggt',
    'amilasi_sierica_u_l': 'serum_amylase',
    'lipasi_sierica_u_l': 'serum_lipase',
    'lattato_deidrogenasi_ldh_u': 'ldh',
    'coledocolitiasi': 'choledocholithiasis',
    'colangite_acuta': 'cholangitis',
    'cpre_con_se_colangiopancre': 'ercp',
    'riscontro_di_recidiva': 'target'
}

# Define the standard order of features (excluding identifiers and target)
STANDARD_FEATURE_ORDER = [
    'age',
    'sex',
    'previous_episodes',
    'admitting_specialty',
    'bmi',
    'diabetes',
    'chronic_pulmonary_disease',
    'hypertension',
    'atrial_fibrillation',
    'ischemic_heart_disease',
    'chronic_kidney_disease',
    'hematopoietic_disease',
    'immunosuppressive_medications',
    'wbc',
    'neutrophils',
    'platelets',
    'inr',
    'crp',
    'ast',
    'alt',
    'total_bilirubin',
    'conjugated_bilirubin',
    'ggt',
    'serum_amylase',
    'serum_lipase',
    'ldh',
    'choledocholithiasis',
    'cholangitis',
    'ercp'
]

# Define categorical and continuous variables (using standardized names)
CATEGORICAL_VARIABLES = [
    'sex',
    'previous_episodes',
    'admitting_specialty',
    'diabetes',
    'chronic_pulmonary_disease',
    'hypertension',
    'atrial_fibrillation',
    'ischemic_heart_disease',
    'chronic_kidney_disease',
    'hematopoietic_disease',
    'immunosuppressive_medications',
    'choledocholithiasis',
    'cholangitis',
    'ercp'
]

CONTINUOUS_VARIABLES = [
    'age',
    'bmi',
    'wbc',
    'neutrophils',
    'platelets',
    'inr',
    'crp',
    'ast',
    'alt',
    'total_bilirubin',
    'conjugated_bilirubin',
    'ggt',
    'serum_amylase',
    'serum_lipase',
    'ldh'
]


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_data(prospective_path, retrospective_path):
    """Load prospective and retrospective datasets"""
    print("\n[1] Loading datasets...")
    prosp = pd.read_excel(prospective_path)
    retro = pd.read_excel(retrospective_path)
    
    print(f"  Prospective: {prosp.shape[0]} samples, {prosp.shape[1]} columns")
    print(f"  Retrospective: {retro.shape[0]} samples, {retro.shape[1]} columns")
    
    # Clean data: handle various missing value representations and comma decimals
    print("  Cleaning data: handling missing values and decimal separators...")
    for df in [prosp, retro]:
        for col in df.columns:
            if df[col].dtype == 'object':
                # Replace common missing value representations with NaN
                df[col] = df[col].replace(['-', '--', 'NA', 'N/A', 'n/a', 'null', ''], np.nan)
                
                # Try to convert strings with comma decimals to floats
                try:
                    df[col] = df[col].apply(lambda x: str(x).replace(',', '.') if pd.notna(x) and isinstance(x, str) else x)
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
    
    return prosp, retro


def standardize_columns(df, dataset_type):
    """
    Standardize column names to common naming convention
    
    Parameters:
    -----------
    df : DataFrame
        Input dataset
    dataset_type : str
        'prospective' or 'retrospective'
    
    Returns:
    --------
    DataFrame with standardized column names
    """
    print(f"\n[2] Standardizing column names for {dataset_type} dataset...")
    
    df_std = df.copy()
    
    if dataset_type == 'prospective':
        # Keep record_id, rename features, rename target
        df_std = df_std.rename(columns=PROSPECTIVE_MAPPING)
        df_std = df_std.rename(columns={'record_id': 'patient_id'})
        
    elif dataset_type == 'retrospective':
        # Rename features and target, drop extra columns
        df_std = df_std.rename(columns=COLUMN_MAPPING)
        
        # Drop extra columns not in prospective
        extra_cols = ['Mortality', 'Early cholecystectomy during index hospital admission']
        df_std = df_std.drop(columns=[col for col in extra_cols if col in df_std.columns], errors='ignore')
        
        # Keep Country and Year as identifiers but rename for consistency
        if 'Country' in df_std.columns:
            df_std = df_std.rename(columns={'Country': 'country'})
        if 'Year of hospital admission for acute biliary pancreatitis' in df_std.columns:
            df_std = df_std.rename(columns={'Year of hospital admission for acute biliary pancreatitis': 'admission_year'})
    
    # Reorder columns: identifiers first, then features in standard order, then target
    id_cols = [col for col in df_std.columns if col not in STANDARD_FEATURE_ORDER + ['target']]
    feature_cols = [col for col in STANDARD_FEATURE_ORDER if col in df_std.columns]
    final_order = id_cols + feature_cols + ['target']
    df_std = df_std[final_order]
    
    print(f"  Standardized columns: {len(df_std.columns)}")
    print(f"  Features: {len(feature_cols)} ({len([c for c in feature_cols if c in CATEGORICAL_VARIABLES])} categorical, {len([c for c in feature_cols if c in CONTINUOUS_VARIABLES])} continuous)")
    
    return df_std


def prepare_dataset(df, dataset_type):
    """Prepare dataset by separating features, target, and identifiers"""
    print(f"\n[3] Preparing {dataset_type} dataset for processing...")
    
    # Identify columns
    if dataset_type == 'prospective':
        id_cols = ['patient_id']
    else:  # retrospective
        id_cols = ['country', 'admission_year']
    
    # Get features (all columns except identifiers and target)
    feature_cols = [col for col in df.columns if col not in id_cols + ['target']]
    
    # Separate components
    X = df[feature_cols].copy()
    y = df['target'].copy()
    identifiers = df[id_cols].copy() if id_cols else None
    
    print(f"  Target variable: 'target'")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Missing values: {X.isnull().sum().sum()} cells ({X.isnull().sum().sum()/(X.shape[0]*X.shape[1])*100:.2f}%)")
    
    return X, y, identifiers, feature_cols


def split_data(X, y, identifiers, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets with stratification
    """
    print("\n[4] Splitting data (Train: 70%, Val: 15%, Test: 15%)...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    if identifiers is not None:
        id_temp, id_test = train_test_split(
            identifiers, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
    else:
        id_temp, id_test = None, None
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_size_adjusted, 
        random_state=random_state, 
        stratify=y_temp
    )
    
    if id_temp is not None:
        id_train, id_val = train_test_split(
            id_temp, 
            test_size=val_size_adjusted, 
            random_state=random_state, 
            stratify=y_temp
        )
    else:
        id_train, id_val = None, None
    
    print(f"  Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    - Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")
    print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    - Class 0: {(y_val==0).sum()}, Class 1: {(y_val==1).sum()}")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"    - Class 0: {(y_test==0).sum()}, Class 1: {(y_test==1).sum()}")
    
    return {
        'train': {'X': X_train, 'y': y_train, 'id': id_train},
        'val': {'X': X_val, 'y': y_val, 'id': id_val},
        'test': {'X': X_test, 'y': y_test, 'id': id_test}
    }


def mice_imputation(X_train, X_val, X_test, random_state=42):
    """
    Perform MICE imputation separately for categorical and continuous variables
    """
    print("\n[5] MICE Imputation...")
    
    X_train_imp = X_train.copy()
    X_val_imp = X_val.copy()
    X_test_imp = X_test.copy()
    
    # Check for missing values
    train_missing = X_train.isnull().sum().sum()
    val_missing = X_val.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    
    print(f"  Missing values before imputation:")
    print(f"    Train: {train_missing}")
    print(f"    Val: {val_missing}")
    print(f"    Test: {test_missing}")
    
    if train_missing == 0 and val_missing == 0 and test_missing == 0:
        print("  No missing values detected. Skipping imputation.")
        return X_train_imp, X_val_imp, X_test_imp, None, None
    
    # Get categorical and continuous variables that exist in the dataset
    cat_vars = [v for v in CATEGORICAL_VARIABLES if v in X_train.columns]
    cont_vars = [v for v in CONTINUOUS_VARIABLES if v in X_train.columns]
    
    # Impute continuous variables
    if cont_vars and X_train[cont_vars].isnull().sum().sum() > 0:
        print(f"  Imputing {len(cont_vars)} continuous variables...")
        imputer_cont = IterativeImputer(
            random_state=random_state,
            max_iter=10,
            verbose=0
        )
        
        X_train_imp[cont_vars] = imputer_cont.fit_transform(X_train[cont_vars])
        X_val_imp[cont_vars] = imputer_cont.transform(X_val[cont_vars])
        X_test_imp[cont_vars] = imputer_cont.transform(X_test[cont_vars])
    else:
        imputer_cont = None
        print("  No missing values in continuous variables.")
    
    # Impute categorical variables
    if cat_vars and X_train[cat_vars].isnull().sum().sum() > 0:
        print(f"  Imputing {len(cat_vars)} categorical variables...")
        
        imputer_cat = IterativeImputer(
            random_state=random_state,
            max_iter=10,
            verbose=0
        )
        
        X_train_imp[cat_vars] = imputer_cat.fit_transform(X_train[cat_vars])
        X_val_imp[cat_vars] = imputer_cat.transform(X_val[cat_vars])
        X_test_imp[cat_vars] = imputer_cat.transform(X_test[cat_vars])
        
        # Round and clip categorical variables
        for col in cat_vars:
            X_train_imp[col] = X_train_imp[col].round()
            X_val_imp[col] = X_val_imp[col].round()
            X_test_imp[col] = X_test_imp[col].round()
            
            min_val = X_train[col].min()
            max_val = X_train[col].max()
            X_train_imp[col] = X_train_imp[col].clip(min_val, max_val)
            X_val_imp[col] = X_val_imp[col].clip(min_val, max_val)
            X_test_imp[col] = X_test_imp[col].clip(min_val, max_val)
    else:
        imputer_cat = None
        print("  No missing values in categorical variables.")
    
    # Verify no missing values remain
    print(f"  Missing values after imputation:")
    print(f"    Train: {X_train_imp.isnull().sum().sum()}")
    print(f"    Val: {X_val_imp.isnull().sum().sum()}")
    print(f"    Test: {X_test_imp.isnull().sum().sum()}")
    
    return X_train_imp, X_val_imp, X_test_imp, imputer_cont, imputer_cat


def normalize_continuous(X_train, X_val, X_test):
    """
    Normalize continuous variables using StandardScaler
    """
    print("\n[6] Normalizing continuous variables...")
    
    cont_vars = [v for v in CONTINUOUS_VARIABLES if v in X_train.columns]
    
    if not cont_vars:
        print("  No continuous variables to normalize.")
        return X_train.copy(), X_val.copy(), X_test.copy(), None
    
    X_train_norm = X_train.copy()
    X_val_norm = X_val.copy()
    X_test_norm = X_test.copy()
    
    print(f"  Normalizing {len(cont_vars)} continuous variables...")
    print(f"  Method: StandardScaler (mean=0, std=1)")
    
    scaler = StandardScaler()
    
    X_train_norm[cont_vars] = scaler.fit_transform(X_train[cont_vars])
    X_val_norm[cont_vars] = scaler.transform(X_val[cont_vars])
    X_test_norm[cont_vars] = scaler.transform(X_test[cont_vars])
    
    print(f"  Training set continuous variables after normalization:")
    print(f"    Mean: {X_train_norm[cont_vars].mean().mean():.6f}")
    print(f"    Std: {X_train_norm[cont_vars].std().mean():.6f}")
    
    return X_train_norm, X_val_norm, X_test_norm, scaler


def save_processed_data(splits, dataset_type, output_dir='processed_data'):
    """Save processed datasets to CSV files"""
    print(f"\n[7] Saving processed {dataset_type} data...")
    
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)
    
    for split_name in ['train', 'val', 'test']:
        split_data = splits[split_name]
        
        # Combine identifiers, features, and target
        df_combined = pd.DataFrame()
        
        # Add identifiers first
        if split_data['id'] is not None:
            for col in split_data['id'].columns:
                df_combined[col] = split_data['id'][col].values
        
        # Add features (already in correct order)
        for col in split_data['X'].columns:
            df_combined[col] = split_data['X'][col].values
        
        # Add target last
        df_combined['target'] = split_data['y'].values
        
        # Save to CSV
        output_path = os.path.join(dataset_dir, f'{split_name}.csv')
        df_combined.to_csv(output_path, index=False)
        print(f"  Saved: {output_path} ({len(df_combined)} samples, {len(df_combined.columns)} columns)")
    
    return dataset_dir


def save_artifacts(imputer_cont, imputer_cat, scaler, dataset_type, output_dir='processed_data'):
    """Save imputers and scaler for future use"""
    print(f"\n[8] Saving preprocessing artifacts for {dataset_type}...")
    
    dataset_dir = os.path.join(output_dir, dataset_type)
    
    if imputer_cont is not None:
        path = os.path.join(dataset_dir, 'imputer_continuous.pkl')
        with open(path, 'wb') as f:
            pickle.dump(imputer_cont, f)
        print(f"  Saved: {path}")
    
    if imputer_cat is not None:
        path = os.path.join(dataset_dir, 'imputer_categorical.pkl')
        with open(path, 'wb') as f:
            pickle.dump(imputer_cat, f)
        print(f"  Saved: {path}")
    
    if scaler is not None:
        path = os.path.join(dataset_dir, 'scaler.pkl')
        with open(path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"  Saved: {path}")

def save_full_processed_dataset(df, dataset_type, output_dir='processed_data'):
    """
    Save full processed dataset (no train/val/test split)
    """
    print(f"\n[7b] Saving full processed {dataset_type} dataset (no split)...")
    
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)
    
    output_path = os.path.join(dataset_dir, 'full_processed.csv')
    df.to_csv(output_path, index=False)
    
    print(f"  Saved: {output_path} ({len(df)} samples, {len(df.columns)} columns)")


def process_dataset(df, dataset_type, output_dir='processed_data'):
    """Complete preprocessing pipeline for a single dataset"""
    print("\n" + "=" * 80)
    print(f"PROCESSING {dataset_type.upper()} DATASET")
    print("=" * 80)
    
    # Standardize columns
    df_std = standardize_columns(df, dataset_type)
    
    # Prepare data
    X, y, identifiers, feature_cols = prepare_dataset(df_std, dataset_type)
    
    # --- NEW: Imputation and scaling on full dataset ---
    print("\n[4] Imputation and normalization on full dataset...")
    
    X_imp, _, _, imputer_cont, imputer_cat = mice_imputation(X, X, X)

    # --- Save dataset after imputation, before normalization ---
    print("\n[4b] Saving full dataset after MICE (no normalization)...")

    df_no_norm = pd.DataFrame()

    # Add identifiers
    if identifiers is not None:
        for col in identifiers.columns:
            df_no_norm[col] = identifiers[col].values

    # Add imputed features
    for col in X_imp.columns:
        df_no_norm[col] = X_imp[col].values

    # Add target
    df_no_norm['target'] = y.values

    # Save to Excel
    dataset_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)

    no_norm_path = os.path.join(dataset_dir, 'full_preprocessed_no_normalized.xlsx')
    df_no_norm.to_excel(no_norm_path, index=False)

    print(f"  Saved: {no_norm_path} ({len(df_no_norm)} samples, {len(df_no_norm.columns)} columns)")

    #richiesta di interruzione
    while True:
        choice = input("\nVuoi continuare con la normalizzazione?\n1. Sì\n2. No (interrompi esecuzione)\nScelta (1/2): ").strip()
    
        if choice == '1':
            print("Continuo con la normalizzazione...")
            break
        elif choice == '2':
            print("Esecuzione interrotta dall'utente dopo l'imputazione.")
            return None, None
        else:
           print("Scelta non valida. Inserisci 1 o 2.")

    #normalization
    X_norm, _, _, scaler = normalize_continuous(X_imp, X_imp, X_imp)
    
    # Rebuild full processed dataframe
    df_full = pd.DataFrame()
    
    if identifiers is not None:
        for col in identifiers.columns:
            df_full[col] = identifiers[col].values
    
    for col in X_norm.columns:
        df_full[col] = X_norm[col].values
    
    df_full['target'] = y.values
    
    # Save full dataset (no split)
    save_full_processed_dataset(df_full, dataset_type, output_dir)
    
    # --- Original split-based pipeline (unchanged) ---
    splits = split_data(X, y, identifiers)
    
    X_train_imp, X_val_imp, X_test_imp, imputer_cont, imputer_cat = mice_imputation(
        splits['train']['X'], 
        splits['val']['X'], 
        splits['test']['X']
    )
    
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_continuous(
        X_train_imp,
        X_val_imp,
        X_test_imp
    )
    
    splits['train']['X'] = X_train_norm
    splits['val']['X'] = X_val_norm
    splits['test']['X'] = X_test_norm
    
    dataset_dir = save_processed_data(splits, dataset_type, output_dir)
    save_artifacts(imputer_cont, imputer_cat, scaler, dataset_type, output_dir)
    
    return splits, dataset_dir



def generate_summary_report(prosp_splits, retro_splits, output_dir='processed_data'):
    """Generate a summary report of the preprocessing"""
    print("\n" + "=" * 80)
    print("PREPROCESSING SUMMARY REPORT")
    print("=" * 80)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATA PREPROCESSING SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Column Standardization:")
    report_lines.append("-" * 40)
    report_lines.append("✓ All column names standardized between datasets")
    report_lines.append("✓ Fixed typo: 'immu0suppressive' → 'immunosuppressive'")
    report_lines.append("✓ Removed extra columns from retrospective: Mortality, Early cholecystectomy")
    report_lines.append("✓ Same 29 features in both datasets, same order")
    report_lines.append("")
    
    for dataset_name, splits in [('Prospective', prosp_splits), ('Retrospective', retro_splits)]:
        report_lines.append(f"\n{dataset_name} Dataset:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total samples: {len(splits['train']['X']) + len(splits['val']['X']) + len(splits['test']['X'])}")
        report_lines.append(f"Features: {splits['train']['X'].shape[1]} (14 categorical, 15 continuous)")
        report_lines.append("")
        report_lines.append(f"Train set: {len(splits['train']['X'])} samples")
        report_lines.append(f"  Class 0: {(splits['train']['y']==0).sum()}, Class 1: {(splits['train']['y']==1).sum()}")
        report_lines.append(f"Validation set: {len(splits['val']['X'])} samples")
        report_lines.append(f"  Class 0: {(splits['val']['y']==0).sum()}, Class 1: {(splits['val']['y']==1).sum()}")
        report_lines.append(f"Test set: {len(splits['test']['X'])} samples")
        report_lines.append(f"  Class 0: {(splits['test']['y']==0).sum()}, Class 1: {(splits['test']['y']==1).sum()}")
        report_lines.append("")
    
    report_lines.append("\nPreprocessing Steps Applied:")
    report_lines.append("1. Column standardization and alignment")
    report_lines.append("2. Data splitting with stratification (70/15/15)")
    report_lines.append("3. MICE imputation for missing values")
    report_lines.append("4. StandardScaler normalization for continuous variables")
    report_lines.append("")
    report_lines.append("Output files saved in:")
    report_lines.append(f"  - {output_dir}/prospective/")
    report_lines.append(f"  - {output_dir}/retrospective/")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save report
    report_path = os.path.join(output_dir, 'preprocessing_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Load datasets
    prospective, retrospective = load_data('raw_data/1_Prospective.xlsx', 'raw_data/2_Retrospective.xlsx')
    
    # Process prospective dataset
    prosp_splits, prosp_dir = process_dataset(prospective, 'prospective')
    
    # Process retrospective dataset
    retro_splits, retro_dir = process_dataset(retrospective, 'retrospective')
    
    # Generate summary report
    generate_summary_report(prosp_splits, retro_splits)
    
    # Verify column alignment
    print("\n" + "=" * 80)
    print("COLUMN ALIGNMENT VERIFICATION")
    print("=" * 80)
    
    prosp_train = pd.read_csv('processed_data/prospective/train.csv')
    retro_train = pd.read_csv('processed_data/retrospective/train.csv')
    
    prosp_features = [c for c in prosp_train.columns if c not in ['patient_id', 'target']]
    retro_features = [c for c in retro_train.columns if c not in ['country', 'admission_year', 'target']]
    
    print(f"\nProspective features ({len(prosp_features)}):")
    for i, col in enumerate(prosp_features, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nRetrospective features ({len(retro_features)}):")
    for i, col in enumerate(retro_features, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n✓ Column names match: {prosp_features == retro_features}")
    print(f"✓ Column order matches: {prosp_features == retro_features}")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print("\nBoth datasets now have:")
    print("  ✓ Same 29 features")
    print("  ✓ Identical column names")
    print("  ✓ Same column order")
    print("  ✓ Standardized values")
    print("=" * 80)
