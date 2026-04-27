"""
Sanity Check: Compare Ranges and Categories Between Prospective and Retrospective
==================================================================================

This script verifies that the processed prospective and retrospective datasets
have compatible value ranges and category distributions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("SANITY CHECK: COMPARING PROSPECTIVE vs RETROSPECTIVE")
print("=" * 80)

# Load all data (combine train/val/test for comprehensive check)
prosp_train = pd.read_csv('processed_data/prospective/train.csv')
prosp_val = pd.read_csv('processed_data/prospective/val.csv')
prosp_test = pd.read_csv('processed_data/prospective/test.csv')
prosp_all = pd.concat([prosp_train, prosp_val, prosp_test], ignore_index=True)

retro_train = pd.read_csv('processed_data/retrospective/train.csv')
retro_val = pd.read_csv('processed_data/retrospective/val.csv')
retro_test = pd.read_csv('processed_data/retrospective/test.csv')
retro_all = pd.concat([retro_train, retro_val, retro_test], ignore_index=True)

# Define variable types
CATEGORICAL_VARIABLES = [
    'sex', 'previous_episodes', 'admitting_specialty', 'diabetes',
    'chronic_pulmonary_disease', 'hypertension', 'atrial_fibrillation',
    'ischemic_heart_disease', 'chronic_kidney_disease', 'hematopoietic_disease',
    'immunosuppressive_medications', 'choledocholithiasis', 'cholangitis', 'ercp'
]

CONTINUOUS_VARIABLES = [
    'age', 'bmi', 'wbc', 'neutrophils', 'platelets', 'inr', 'crp',
    'ast', 'alt', 'total_bilirubin', 'conjugated_bilirubin', 'ggt',
    'serum_amylase', 'serum_lipase', 'ldh'
]

# Extract features only (remove identifiers and target)
prosp_features = prosp_all[[col for col in prosp_all.columns 
                            if col not in ['patient_id', 'target']]]
retro_features = retro_all[[col for col in retro_all.columns 
                            if col not in ['country', 'admission_year', 'target']]]

print(f"\nProspective: {len(prosp_all)} total samples")
print(f"Retrospective: {len(retro_all)} total samples")

# =============================================================================
# CHECK 1: CATEGORICAL VARIABLES
# =============================================================================
print("\n" + "=" * 80)
print("CHECK 1: CATEGORICAL VARIABLES")
print("=" * 80)

categorical_issues = []

for var in CATEGORICAL_VARIABLES:
    print(f"\n{var}:")
    print("-" * 40)
    
    prosp_cats = sorted(prosp_features[var].unique())
    retro_cats = sorted(retro_features[var].unique())
    
    print(f"  Prospective categories: {prosp_cats}")
    print(f"  Retrospective categories: {retro_cats}")
    
    # Check if categories match
    if set(prosp_cats) == set(retro_cats):
        print(f"  ✓ Categories match perfectly")
    else:
        # Check if one is subset of other (acceptable)
        prosp_set = set(prosp_cats)
        retro_set = set(retro_cats)
        
        if prosp_set.issubset(retro_set):
            print(f"  ⚠ Prospective is subset of Retrospective")
            print(f"    Extra in Retro: {retro_set - prosp_set}")
        elif retro_set.issubset(prosp_set):
            print(f"  ⚠ Retrospective is subset of Prospective")
            print(f"    Extra in Prosp: {prosp_set - retro_set}")
        else:
            print(f"  ❌ Categories differ significantly!")
            print(f"    Only in Prosp: {prosp_set - retro_set}")
            print(f"    Only in Retro: {retro_set - prosp_set}")
            categorical_issues.append(var)
    
    # Show distribution
    prosp_dist = prosp_features[var].value_counts(normalize=True).sort_index()
    retro_dist = retro_features[var].value_counts(normalize=True).sort_index()
    
    print(f"  Prospective distribution: {dict(prosp_dist.round(3))}")
    print(f"  Retrospective distribution: {dict(retro_dist.round(3))}")

# =============================================================================
# CHECK 2: CONTINUOUS VARIABLES (NORMALIZED)
# =============================================================================
print("\n" + "=" * 80)
print("CHECK 2: CONTINUOUS VARIABLES (After Normalization)")
print("=" * 80)
print("\nNote: These are normalized, so ranges should be roughly similar")
print("We check if distributions are reasonably comparable.\n")

continuous_issues = []

for var in CONTINUOUS_VARIABLES:
    print(f"\n{var}:")
    print("-" * 40)
    
    prosp_stats = {
        'min': prosp_features[var].min(),
        'q25': prosp_features[var].quantile(0.25),
        'median': prosp_features[var].median(),
        'q75': prosp_features[var].quantile(0.75),
        'max': prosp_features[var].max(),
        'mean': prosp_features[var].mean(),
        'std': prosp_features[var].std()
    }
    
    retro_stats = {
        'min': retro_features[var].min(),
        'q25': retro_features[var].quantile(0.25),
        'median': retro_features[var].median(),
        'q75': retro_features[var].quantile(0.75),
        'max': retro_features[var].max(),
        'mean': retro_features[var].mean(),
        'std': retro_features[var].std()
    }
    
    print(f"  Prospective: min={prosp_stats['min']:.2f}, q25={prosp_stats['q25']:.2f}, "
          f"median={prosp_stats['median']:.2f}, q75={prosp_stats['q75']:.2f}, max={prosp_stats['max']:.2f}")
    print(f"  Retrospective: min={retro_stats['min']:.2f}, q25={retro_stats['q25']:.2f}, "
          f"median={retro_stats['median']:.2f}, q75={retro_stats['q75']:.2f}, max={retro_stats['max']:.2f}")
    
    # Check if ranges overlap significantly
    prosp_range = (prosp_stats['q25'], prosp_stats['q75'])
    retro_range = (retro_stats['q25'], retro_stats['q75'])
    
    # Calculate overlap
    overlap_start = max(prosp_range[0], retro_range[0])
    overlap_end = min(prosp_range[1], retro_range[1])
    
    if overlap_end > overlap_start:
        overlap_pct = (overlap_end - overlap_start) / (prosp_range[1] - prosp_range[0]) * 100
        print(f"  ✓ IQR overlap: {overlap_pct:.1f}%")
    else:
        print(f"  ⚠ No IQR overlap - distributions may be quite different")
        continuous_issues.append(var)
    
    # Check for extreme differences in spread
    std_ratio = max(prosp_stats['std'], retro_stats['std']) / min(prosp_stats['std'], retro_stats['std'])
    if std_ratio > 2:
        print(f"  ⚠ Large difference in spread (std ratio: {std_ratio:.2f})")

# =============================================================================
# CHECK 3: TARGET VARIABLE
# =============================================================================
print("\n" + "=" * 80)
print("CHECK 3: TARGET VARIABLE")
print("=" * 80)

prosp_target_dist = prosp_all['target'].value_counts(normalize=True).sort_index()
retro_target_dist = retro_all['target'].value_counts(normalize=True).sort_index()

print(f"\nProspective target distribution:")
print(f"  Class 0: {prosp_target_dist[0]:.1%} ({(prosp_all['target']==0).sum()} samples)")
print(f"  Class 1: {prosp_target_dist[1]:.1%} ({(prosp_all['target']==1).sum()} samples)")

print(f"\nRetrospective target distribution:")
print(f"  Class 0: {retro_target_dist[0]:.1%} ({(retro_all['target']==0).sum()} samples)")
print(f"  Class 1: {retro_target_dist[1]:.1%} ({(retro_all['target']==1).sum()} samples)")

target_diff = abs(prosp_target_dist[1] - retro_target_dist[1])
print(f"\nClass imbalance difference: {target_diff:.1%}")
if target_diff < 0.05:
    print("✓ Target distributions are very similar")
elif target_diff < 0.10:
    print("✓ Target distributions are reasonably similar")
else:
    print("⚠ Target distributions differ notably (but may still be acceptable)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SANITY CHECK SUMMARY")
print("=" * 80)

print(f"\n✓ Total features checked: {len(CATEGORICAL_VARIABLES) + len(CONTINUOUS_VARIABLES)}")
print(f"  - Categorical: {len(CATEGORICAL_VARIABLES)}")
print(f"  - Continuous: {len(CONTINUOUS_VARIABLES)}")

if categorical_issues:
    print(f"\n⚠ Categorical variables with significant differences: {len(categorical_issues)}")
    for var in categorical_issues:
        print(f"  - {var}")
else:
    print(f"\n✓ All categorical variables have compatible categories")

if continuous_issues:
    print(f"\n⚠ Continuous variables with no IQR overlap: {len(continuous_issues)}")
    for var in continuous_issues:
        print(f"  - {var}")
else:
    print(f"\n✓ All continuous variables have overlapping distributions")

# =============================================================================
# DETAILED COMPARISON FOR CRITICAL VARIABLES
# =============================================================================
print("\n" + "=" * 80)
print("DETAILED CHECK: CRITICAL BINARY VARIABLES")
print("=" * 80)

binary_vars = ['sex', 'previous_episodes', 'diabetes', 'chronic_pulmonary_disease',
               'hypertension', 'atrial_fibrillation', 'ischemic_heart_disease',
               'chronic_kidney_disease', 'hematopoietic_disease', 
               'immunosuppressive_medications', 'cholangitis']

print("\nChecking if binary variables (0/1) are coded consistently:\n")

all_binary_ok = True
for var in binary_vars:
    prosp_vals = set(prosp_features[var].unique())
    retro_vals = set(retro_features[var].unique())
    
    # Check if values are subset of {0, 1}
    prosp_is_binary = prosp_vals.issubset({0, 1, 0.0, 1.0})
    retro_is_binary = retro_vals.issubset({0, 1, 0.0, 1.0})
    
    if prosp_is_binary and retro_is_binary:
        print(f"  ✓ {var}: Both are binary (0/1)")
    else:
        print(f"  ❌ {var}: Not properly binary!")
        print(f"     Prosp values: {prosp_vals}")
        print(f"     Retro values: {retro_vals}")
        all_binary_ok = False

if all_binary_ok:
    print("\n✓ All binary variables are properly coded!")

# =============================================================================
# FINAL VERDICT
# =============================================================================
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

if not categorical_issues and not continuous_issues and all_binary_ok:
    print("\n✅ PASS: Both datasets are compatible and ready for modeling!")
    print("   - All categorical variables have compatible categories")
    print("   - All continuous variables have reasonable distributions")
    print("   - Binary coding is consistent")
elif len(categorical_issues) + len(continuous_issues) <= 2:
    print("\n⚠️ ACCEPTABLE: Minor differences detected but likely not problematic")
    print("   - Most variables are compatible")
    print("   - Small differences may reflect genuine data collection variations")
else:
    print("\n❌ CAUTION: Significant differences detected")
    print("   - Review the flagged variables before modeling")
    print("   - Consider if differences reflect data quality or genuine variation")

print("\n" + "=" * 80)
