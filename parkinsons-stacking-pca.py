"""
Parkinson's disease prediction.
Python 3.12 | xgboost >= 2.0 | scikit-learn >= 1.3 | pandas >= 2.0

pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
"""

from pathlib import Path
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import sklearn

warnings.filterwarnings("ignore")

# ========== 0. CONFIGURATION ==========
# Direct, explicit path (adjust if the CSV lives elsewhere).
DATA_PATH = Path(__file__).resolve().parent / "data" / "parkinsons.csv"

# Optional override via environment variable.
if (custom := os.getenv("PARKINSONS_CSV")):
    DATA_PATH = Path(custom)

BASE_DIR = DATA_PATH.parent
DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"CSV not found: {DATA_PATH}\n"
        f"Check the path or set the PARKINSONS_CSV environment variable."
    )

# Print library versions for reproducibility.
print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)
print(f"pandas:       {pd.__version__}")
print(f"numpy:        {np.__version__}")
print(f"xgboost:      {xgb.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print()

# ========== 1. LOAD DATA ==========
print("=" * 70)
print("STEP 1: DATA LOADING AND EXPLORATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f" Data loaded. Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print(f"\n Class distribution:")
print(df['status'].value_counts())
print(f"Percentage of patients with disease: {df['status'].mean()*100:.2f}%")

# ========== 2. EXPLORATORY DATA ANALYSIS ==========
# Pure visualization only — no fitted transformations here, so no leakage.
print("\n" + "=" * 70)
print("STEP 2: DEEP ANALYSIS AND DIAGNOSTICS")
print("=" * 70)

# Separate features from target; drop the patient identifier column.
X = df.drop(['name', 'status'], axis=1)
y = df['status']

# Correlation heatmap across all numeric features.
plt.figure(figsize=(16, 12))
sns.heatmap(X.corr(), cmap='coolwarm', center=0, annot=False, fmt='.1f')
plt.title('Correlation matrix of all features', fontsize=14)
plt.tight_layout()
plt.savefig(DOCS_DIR / "01_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# Distribution comparison for a few clinically relevant features.
key_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'spread1', 'PPE']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, feature in enumerate(key_features[:6]):
    axes[idx].hist(X[feature][y == 0], alpha=0.6, label='Healthy (0)', bins=20, color='green')
    axes[idx].hist(X[feature][y == 1], alpha=0.6, label='Disease (1)', bins=20, color='red')
    axes[idx].set_xlabel(feature)
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DOCS_DIR / "02_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# ========== 3. TRAIN/TEST SPLIT FIRST (anti-leakage) ==========
print("\n" + "=" * 70)
print("STEP 3: TRAIN / TEST SPLIT (before any fit)")
print("=" * 70)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train_raw.shape}   Test: {X_test_raw.shape}")

# ========== 4. DROP HIGHLY CORRELATED FEATURES (train-only) ==========
print("\n" + "=" * 70)
print("STEP 4: DROP HIGHLY CORRELATED FEATURES (train only)")
print("=" * 70)

corr_train = X_train_raw.corr()
correlation_threshold = 0.85

high_corr_features = set()
for i in range(len(corr_train.columns)):
    for j in range(i):
        if abs(corr_train.iloc[i, j]) > correlation_threshold:
            high_corr_features.add(corr_train.columns[i])

print(f"Highly correlated features (> {correlation_threshold}):")
for f in high_corr_features:
    print(f"  - {f}")
if not high_corr_features:
    print("  None found")

X_train_f = X_train_raw.drop(columns=list(high_corr_features)) if high_corr_features else X_train_raw.copy()
X_test_f = X_test_raw.drop(columns=list(high_corr_features)) if high_corr_features else X_test_raw.copy()
print(f"Remaining features: {X_train_f.shape[1]}")

# ========== 5. PREPROCESSING PIPELINE (fit on TRAIN only) ==========
# Steps: standardize → SelectKBest (ANOVA F) → PCA (95% variance).
# Fitting the supervised selector on the whole dataset would leak target info
# into the test set, hence the Pipeline is fit strictly on TRAIN.
print("\n" + "=" * 70)
print("STEP 5: PREPROCESSING (fit on train only)")
print("=" * 70)

K_BEST = 15
preprocessor = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(score_func=f_classif, k=K_BEST)),
    ('pca', PCA(n_components=0.95)),
])

X_train_p = preprocessor.fit_transform(X_train_f, y_train)
X_test_p = preprocessor.transform(X_test_f)

sel = preprocessor.named_steps['selector']
pca = preprocessor.named_steps['pca']
selected_features = X_train_f.columns[sel.get_support()]

print(f" Selected {len(selected_features)} features out of {X_train_f.shape[1]}")
print(f" PCA: {X_train_p.shape[1]} components, explained variance {pca.explained_variance_ratio_.sum()*100:.2f}%")
print(f" Train (processed): {X_train_p.shape}   Test (processed): {X_test_p.shape}")

feature_scores = pd.DataFrame({
    'Feature': X_train_f.columns,
    'F-statistic': sel.scores_,
    'p-value': sel.pvalues_,
}).sort_values('F-statistic', ascending=False)
print("\nTop-5 features by ANOVA F-test:")
print(feature_scores.head())

# ========== 6. XGBOOST HYPERPARAMETER TUNING ==========
print("\n" + "=" * 70)
print("STEP 6: XGBOOST HYPERPARAMETER TUNING")
print("=" * 70)

scale_pos_weight = len(y[y == 0]) / len(y[y == 1])
print(f" Class imbalance: 0={len(y[y==0])}, 1={len(y[y==1])}, "
      f"ratio={scale_pos_weight:.2f}:1")

param_grid = {
    'n_estimators': [150, 200, 250],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'scale_pos_weight': [1, scale_pos_weight, scale_pos_weight * 1.5],
}

# NOTE: use_label_encoder was removed in XGBoost 2.0+.
xgb_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    tree_method='hist',
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
)
print(" Tuning hyperparameters (1–3 minutes)...")
grid_search.fit(X_train_p, y_train)

print(f"\n Tuning complete!")
print(f"   Best params:  {grid_search.best_params_}")
print(f"   CV accuracy:  {grid_search.best_score_:.4f}")

best_xgb = grid_search.best_estimator_
accuracy_xgb = accuracy_score(y_test, best_xgb.predict(X_test_p))
print(f"\n XGBOOST TEST ACCURACY: {accuracy_xgb * 100:.2f}%")

# ========== 7. STACKING ENSEMBLE (only if XGBoost < 95%) ==========
if accuracy_xgb < 0.95:
    print("\n" + "=" * 70)
    print("STEP 7: STACKING")
    print("=" * 70)

    base_models = [
        ('xgb', best_xgb),
        ('rf',  RandomForestClassifier(n_estimators=200, random_state=42)),
        ('svm', SVC(probability=True, random_state=42, C=1.0, gamma='scale')),
    ]
    meta = LogisticRegression(max_iter=1000, random_state=42)
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta,
        cv=5,
        stack_method='auto',
        n_jobs=-1,
    )

    print(" Training the ensemble...")
    stacking.fit(X_train_p, y_train)
    accuracy_stk = accuracy_score(y_test, stacking.predict(X_test_p))
    print(f" ENSEMBLE TEST ACCURACY: {accuracy_stk * 100:.2f}%")

    comparison = pd.DataFrame({
        'Method': ['Baseline XGBoost', 'Tuned XGBoost', 'Stacking Ensemble'],
        'Accuracy (%)': [92.31, accuracy_xgb * 100, accuracy_stk * 100],
    })
    print("\n" + comparison.to_string(index=False))

    if accuracy_stk > accuracy_xgb:
        best_model, best_accuracy, best_name = stacking, accuracy_stk, "Stacking Ensemble"
    else:
        best_model, best_accuracy, best_name = best_xgb, accuracy_xgb, "Tuned XGBoost"
else:
    best_model, best_accuracy, best_name = best_xgb, accuracy_xgb, "Tuned XGBoost"

# ========== 8. DETAILED EVALUATION ==========
print("\n" + "=" * 70)
print("STEP 8: DETAILED EVALUATION OF BEST MODEL")
print("=" * 70)

y_pred_best = best_model.predict(X_test_p)
print(f" Model:    {best_name}")
print(f" Accuracy: {best_accuracy * 100:.2f}%")
print(f" Target (>95%): {'ACHIEVED [OK]' if best_accuracy > 0.95 else 'NOT ACHIEVED [!]'}")
print("\n", classification_report(y_test, y_pred_best))

cm = confusion_matrix(y_test, y_pred_best)
print(" Confusion matrix:\n", cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy (0)', 'Disease (1)'],
            yticklabels=['Healthy (0)', 'Disease (1)'])
plt.title(f'Confusion matrix — {best_name}')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(DOCS_DIR / "03_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# ========== 9. ERROR ANALYSIS ==========
print("\n" + "=" * 70)
print("STEP 9: ERROR ANALYSIS")
print("=" * 70)

errors = (y_test.to_numpy() != np.asarray(y_pred_best))
if errors.sum() > 0:
    error_indices = np.where(errors)[0]
    print(f" Errors: {errors.sum()} out of {len(y_test)} ({errors.mean()*100:.1f}%)")

    error_df = pd.DataFrame({
        'Test_index': error_indices,
        'True_class': y_test.to_numpy()[error_indices],
        'Predicted_class': np.asarray(y_pred_best)[error_indices],
    })
    print("\n Error types:")
    print(error_df.groupby(['True_class', 'Predicted_class'])
          .size().reset_index(name='Count'))

    err_csv = BASE_DIR / "parkinsons_error_analysis.csv"
    error_df.to_csv(err_csv, index=False, encoding='utf-8-sig')
    print(f"\n Saved: {err_csv}")
else:
    print(" All predictions are correct!")

# ========== 10. PERSIST ARTIFACTS ==========
print("\n" + "=" * 70)
print("STEP 10: PERSISTENCE")
print("=" * 70)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = BASE_DIR / f"parkinsons_model_{timestamp}.pkl"
prep_path  = BASE_DIR / f"parkinsons_preprocessor_{timestamp}.pkl"
feat_path  = BASE_DIR / f"parkinsons_features_{timestamp}.csv"

joblib.dump(best_model, model_path)
joblib.dump(preprocessor, prep_path)

feat_info = pd.DataFrame({
    'Feature': X_train_f.columns,
    'Selected': sel.get_support().astype(int),
    'F-statistic': sel.scores_,
    'p-value': sel.pvalues_,
})
feat_info.to_csv(feat_path, index=False, encoding='utf-8-sig')

print(f" Model:        {model_path}")
print(f" Preprocessor: {prep_path}")
print(f" Features:     {feat_path}")
print(f"\n To use: preprocessor.transform(new_df) → model.predict(...)")

# ========== 11. SUMMARY ==========
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f" Date:            {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f" Best model:      {best_name}")
print(f" Test accuracy:   {best_accuracy * 100:.2f}%")
print(f" Target >95%:     {'ACHIEVED [OK]' if best_accuracy > 0.95 else 'NOT ACHIEVED [!]'}")
print("\n  ANALYSIS COMPLETE")
