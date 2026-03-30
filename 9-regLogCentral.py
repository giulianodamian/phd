# %%
import os
import traceback
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
try:
    matplotlib.use('Agg')
except ImportError:
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        print("Warning: No suitable Matplotlib backend found ('Agg' or 'TkAgg'). Plots may not work.")
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from imblearn.over_sampling import SMOTE
import warnings
from io import StringIO
import re

print('Libraries for imported logistics analysis!')



# %%
# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='imblearn')

# %%
INPUT_TXT_FILE='/scratch/users/giuliano.damian/text/df_code_8.txt'

print(f'Loading data from file "{INPUT_TXT_FILE}" ---')
final_df=pd.DataFrame()
try:
    final_df=pd.read_csv(INPUT_TXT_FILE, sep='\t', encoding='utf-8')
    if not final_df.empty:
        print(f'DataFrame loaded successfully from "{INPUT_TXT_FILE}". Dimensions: {final_df.shape}')
    else:
        print(f'File "{INPUT_TXT_FILE}" was read, but is empty or did not became an DataFrame.')
except FileNotFoundError:
    print(f"ERROR: data file '{INPUT_TXT_FILE}' not found.")
except Exception as e:
    print(f"ERROR reanding file '{INPUT_TXT_FILE}': {e}")
    traceback.print_exc()

# %%
class Config:
    INPUT_DESCRIPTION=f'Data read from {INPUT_TXT_FILE}'
    MAX_ITERATION=1000
    MAX_CORR=0.7
    VIF_THRESHOLD= 10.0
    TARGET='AGN_ionization'
    ALPHA=0.05
    PREDICTION_THRESHOLD=0.5
    COLUMNS_TO_IGNORE=['source_file', 'x', 'y']
    SMOTE_SAMPLING_STRATEGY='auto'
    SMOTE_K_NEIGHBORS=5
    SMOTE_RANDOM_STATE=42
    APPLY_SMOTE=True
    OUTPUT_DIR='/scratch/users/giuliano.damian/text/9-final_analysis_df_output_default'
    OUTPUT_TXT=os.path.join(OUTPUT_DIR, '9-final_result_analysis.txt')
# %%
def save_to_txt(content, file_path, mode='a'):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode, encoding='utf-8') as f:
            f.write(str(content) + '\n')
    except Exception as e:
        print(f'ERROR saving on {file_path}: {e}')

# %%
def run_vif_analysis(X_input, vif_threshold, section_title='VIF Analysis'):

    save_to_txt(f'\n\n=== {section_title} ===', Config.OUTPUT_TXT)
    save_to_txt(f'Iteratively removing variables with VIF > {vif_threshold}', Config.OUTPUT_TXT)

    if X_input.empty:
        save_to_txt("VIF WARNING: Input DataFrame is empty. No VIF analysis performed.", Config.OUTPUT_TXT)
        return pd.DataFrame(), [], []

    X_numeric=X_input.select_dtypes(include=np.number)
    non_numeric_cols=X_input.columns.difference(X_numeric.columns)
    if not non_numeric_cols.empty:
        save_to_txt(f"VIF WARNING: Non-numeric columns found and ignored in the VIF analysis: {', '.join(non_numeric_cols)}", Config.OUTPUT_TXT)

    if X_numeric.empty:
        save_to_txt('VIF WARNING: No numeric column for VIF analysis.', Config.OUTPUT_TXT)    
        return pd.DataFrame(), [], []
    
    X_vif=X_numeric.copy()
    removed_vars_step=[]
    iteration=0
    initial_cols_count=X_vif.shape[1]

    # Removing columns with 0 variance, they cause problems on VIF
    zero_variance_cols=X_vif.columns[X_vif.var(skipna=True) == 0]
    if len(zero_variance_cols) > 0:
        save_to_txt(f"Removing {len(zero_variance_cols)} columns with zero variance: {', '.join(zero_variance_cols)}", Config.OUTPUT_TXT)
        X_vif=X_vif.drop(columns=zero_variance_cols)
        removed_vars_step.extend([(col, 'Zero Variance') for col in zero_variance_cols])
    
    # Removing all NaN columns
    nan_cols_in_vif_input=X_vif.columns[X_vif.isna().all()]
    if len(nan_cols_in_vif_input) > 0:
        save_to_txt(f"VIF WARNING: {X_vif.shape[1]} remaining numeric column(s) after removing zero/NaN variables. VIF not applicable or trivial.", Config.OUTPUT_TXT)
        X_vif=X_vif.drop(columns=nan_cols_in_vif_input)
        removed_vars_step.extend([(col, 'All NaN (VIF)') for col in nan_cols_in_vif_input])
    
    # If only one or no columns remain, VIF is not applicable or is trivial.
    if X_vif.shape[1] <= 1:
        save_to_txt(f"VIF WARNING: {X_vif.shape[1]} remaining numeric column(s) after removing zero/NaN variables. VIF not applicable or trivial.", Config.OUTPUT_TXT)
        final_vars_vif=X_vif.columns.tolist()
        return X_input[final_vars_vif].copy() if final_vars_vif and not X_input.empty and all(c in X_input.columns for c in final_vars_vif) else pd.DataFrame(), final_vars_vif, removed_vars_step
    
    # Main Loop for iterative VIF remotion
    while X_vif.shape[1] > 1:
        iteration += 1
        vif_data=pd.DataFrame()
        vif_data['Variable']=X_vif.columns
        try:
            # Fill in any remaining NaNs in the iteration (if any) to avoid errors in the VIF.
            if X_vif.isnull().values.any():
                nan_cols_loop=X_vif.columns[X_vif.isnull().any()].tolist()
                save_to_txt(f"VIF WARNING (unexpected): NaNs still found in: {nan_cols_loop}. Filling with the average.", Config.OUTPUT_TXT)
                for col_fill_nan in nan_cols_loop:
                    X_vif[col_fill_nan]=X_vif[col_fill_nan].fillna(X_vif[col_fill_nan].mean(skipna=True))

            # Double-check after filling in if there are still any persistent NaNs.
            if X_vif.isnull().values.any():
                cols_with_persistent_nan=X_vif.columns[X_vif.isnull().any()].tolist()
                save_to_txt(f"VIF WARNING: Persistent NaNs in {cols_with_persistent_nan} after fillna (average may be NaN). Removing these columns.", Config.OUTPUT_TXT)
                X_vif=X_vif.drop(columns=cols_with_persistent_nan)
                removed_vars_step.extend([(col, 'Persistent NaN') for col in cols_with_persistent_nan])
                if X_vif.shape[1] <= 1: break
                continue

            # Calculate the VIF values.
            vif_values=[variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
            vif_data['VIF']=vif_values
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
            save_to_txt(f"ERROR in VIF calculation in iteration {iteration}: {e}", Config.OUTPUT_TXT)

            # Attempts to heuristically remove the variable with the highest VIF in case of an error.
            if not vif_data.empty and 'VIF' in vif_data.columns and not vif_data['VIF'].isnull().all():
                vif_data_sorted_err=vif_data.sort_values('VIF', ascending=False, na_position='last').reset_index(drop=True)
                if not vif_data_sorted_err.empty:
                    var_to_remove_on_error=vif_data_sorted_err['Variable'].iloc[0]
                    save_to_txt(f"Trying to remove heuristically: '{var_to_remove_on_error}'", Config.OUTPUT_TXT)
                    if var_to_remove_on_error in X_vif.columns:
                        X_vif=X_vif.drop(columns=[var_to_remove_on_error])
                        removed_vars_step.append((var_to_remove_on_error, f"REmoved for VIF Error ({type(e).__name__})"))
                        continue
                    else: save_to_txt("It was not possible to remove the variable heuristically.", Config.OUTPUT_TXT)
            save_to_txt("Interrupting VIF analysis due to an error.", Config.OUTPUT_TXT)
            break
        except Exception as e:
            save_to_txt(f"UNEXPECTED ERROR in VIF calculation iteration {iteration}: {e}", Config.OUTPUT_TXT)
            save_to_txt(f"Traceback of unexpected VIF error:\n{traceback.format_exc()}", Config.OUTPUT_TXT)
            break
    
        # Sorts VIFs and removes the variable with the largest VIF.
        vif_data=vif_data.sort_values('VIF', ascending=False, na_position='last').reset_index(drop=True)
        if vif_data.empty: break

        max_vif=vif_data['VIF'].iloc[0]
        var_to_remove=vif_data['Variable'].iloc[0]

        # Stop condition: if the maximum VIF is below the threshold.
        if pd.isna(max_vif) or np.isinf(max_vif):
            reason_removal='NaN/Inf VIF'
        elif max_vif <= vif_threshold:
            save_to_txt(f"VIF process completed (Iter {iteration}). Maximum VIF={max_vif:.4f} <= {vif_threshold}", Config.OUTPUT_TXT)
            break
        else:
            reason_removal=max_vif

        # Remove the variable with the highest VIF.
        X_vif=X_vif.drop(columns=[var_to_remove])
        removed_vars_step.append((var_to_remove, reason_removal))

        # Safety limit to prevent infinite loops.
        if iteration > initial_cols_count * 2 + 20:
            save_to_txt("WARNING: VIF process has reached iteration limit. Aborting.", Config.OUTPUT_TXT)
            break
    
    # Log of removed variables
    if X_vif.shape[1] <= 1 and iteration > 0:
        save_to_txt(f" - WARNING: VIF process resulted in {X_vif.shape[1]} remaining variable(s).", Config.OUTPUT_TXT)

    if removed_vars_step:
        save_to_txt("\n Summary of variables removed in this VIF step:", Config.OUTPUT_TXT)
        for var, vif_val_or_reason_str in removed_vars_step:
            if isinstance(vif_val_or_reason_str, str): reason_text=vif_val_or_reason_str
            elif pd.isna(vif_val_or_reason_str): reason_text="VIF=NaN"
            elif np.isinf(vif_val_or_reason_str): reason_text="VIF=Inf"
            else: reason_text=f"VIF={vif_val_or_reason_str:.4f}"
            save_to_txt(f"  - {var}: {reason_text}", Config.OUTPUT_TXT)
    else:
        save_to_txt("No variables were removed in this VIF step.", Config.OUTPUT_TXT)
    
    final_vars_list=X_vif.columns.tolist()
    save_to_txt(f"\n Remaining variables after this VIF step: {len(final_vars_list)}", Config.OUTPUT_TXT)

    return X_input[final_vars_list].copy() if final_vars_list and not X_input.empty and all(c in X_input.columns for c in final_vars_list) else pd.DataFrame(), final_vars_list, removed_vars_step


# %%
def generate_message_significance_ic_or(lower_ci_or_preciso, upper_ci_or_preciso, for_log_remotion=False):
    precisely_have_one=False
    if not(pd.isna(lower_ci_or_preciso) or pd.isna(upper_ci_or_preciso)):
        precisely_have_one=(lower_ci_or_preciso <= 1.0 <= upper_ci_or_preciso)

    s_lower_display_formatted=f"{lower_ci_or_preciso:.4f}"
    s_upper_display_formatted=f"{upper_ci_or_preciso: .4f}"

    if pd.isna(lower_ci_or_preciso) or pd.isna(upper_ci_or_preciso):
        return "IC Unavailable"
    
    if precisely_have_one:
        # 1.0 is genuinely within the range (not statistically significant).
        if for_log_remotion:
            return f"IC from OR [{s_lower_display_formatted}, {s_upper_display_formatted}] contains 1.0."
        else:
            return "Not Significant (OR confidence interval contains 1.0)"
    else:
        # 1.0 is genuinely NOT in the range (statistically significant)
        # Checks if the formatted display creates the visual contradiction of appearing as [1.0000, 1.0000]
        # This happens when the limits are very close to 1.0, but do not contain it.
        if (float(s_lower_display_formatted) == 1.0000) and (float(s_upper_display_formatted) == 1.0000):
            if for_log_remotion:
                return f"IC from OR [{s_lower_display_formatted}, {s_upper_display_formatted}] does NOT contain 1.0 (limits round up to 1.0000 due to extreme proximity)."
            else:
                return "Significant (CI does NOT contain 1.0; limits rounded to 1.0000)"
        else:
            # No visual contradiction And it is significant
            if for_log_remotion:
                return f"IC from OR [{s_lower_display_formatted}, {s_upper_display_formatted}] does NOT contain 1.0."
            else:
                return "Significant (IC of OR does NOT contain 1.0)"

# %%
def execute_logistics_analysis(dataframe_input, apply_smote, filename_suffix):

    print(f"\n--- STARTING LOGISTICS ANALYSIS ({filename_suffix.replace('_', ' ').strip().upper()}) ---")

    Config.APPLY_SMOTE=apply_smote
    Config.OUTPUT_DIR=f"analysis_output{filename_suffix}"
    Config.OUTPUT_TXT=os.path.join(Config.OUTPUT_DIR, f"analysis_result{filename_suffix}.txt")

    print(f"Script settings for this execution ({filename_suffix.replace('_', ' ').strip().upper()}):")
    print(f" - Apply SMOTE: {Config.APPLY_SMOTE}")
    print(f" - Output Directory: {Config.OUTPUT_DIR}")
    print(f" - Report File: {Config.OUTPUT_TXT}")

    script_start_time=pd.Timestamp.now()
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    report_header=f"LOGISTICS ANALYSIS REPORT ({filename_suffix.replace('_', ' ').strip().upper()})\n"
    report_header += f"Source Data: {Config.INPUT_DESCRIPTION}\n" 
    report_header += f"Analysis Date: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_header += f"SMOTE Applied: {'Yes' if Config.APPLY_SMOTE else 'No'}"
    report_header += '=' * 60 + '\n\n'
    save_to_txt(report_header, Config.OUTPUT_TXT, mode='w')

    # --- INITIAL DATA PREPARATION ---

    save_to_txt("=== INITIAL DATA PREPARATION (FROM `dataframe_input`) ===", Config.OUTPUT_TXT)
    df=dataframe_input.copy()
    cols_to_drop=[col for col in Config.COLUMNS_TO_IGNORE if col in df.columns]
    if cols_to_drop:
        df=df.drop(columns=cols_to_drop)
        save_to_txt(f"Ignored columns removed: {', '.join(cols_to_drop)}", Config.OUTPUT_TXT)
    else:
        save_to_txt(f"None of the columns to ignore ({', '.join(Config.COLUMNS_TO_IGNORE)}) were found.", Config.OUTPUT_TXT)
    save_to_txt(f"Dimensions after removal (BEFORE dropna): {df.shape}", Config.OUTPUT_TXT)

    # Clears column names before checking the target
    # Regex to remove special characters, except letters, numbers, and underscores
    df.columns=[re.sub(r'[^a-z0-9_]+', '', col.strip().lower().replace('.', '_')) for col in df.columns]

    # Ensures the target's name is also clear for comparison.
    cleaned_target_name=re.sub(r'[^a-z0-9_]+', '', Config.TARGET.lower().replace('.', '_'))
    current_target_column_name_in_df=cleaned_target_name 
    save_to_txt(f"\nCleaned column names. Expected target: '{current_target_column_name_in_df}'", Config.OUTPUT_TXT)

    # Checks if the target column exists after cleaning the names.
    if current_target_column_name_in_df not in df.columns:
        found_target=False
        for col in df.columns:
            if current_target_column_name_in_df in col:
                df.rename(columns={col: current_target_column_name_in_df}, inplace=True)
                save_to_txt(f"WARNING: Original target '{Config.TARGET}' not found, but '{col}' has been renamed to '{current_target_column_name_in_df}'.", Config.OUTPUT_TXT)
                found_target=True
                break
        if not found_target:
            msg=f"CRITICAL ERROR: Target '{Config.TARGET}' or '{current_target_column_name_in_df}' not found in the DataFrame."
            print(msg)
            save_to_txt(msg, Config.OUTPUT_TXT)
            return
    
    lines_before_dropna=len(df)
    df.dropna(inplace=True)
    lines_after_dropna=len(df)

    removed_nan_lines=lines_before_dropna - lines_after_dropna

    if removed_nan_lines > 0:
        save_to_txt(f"\nREMOVAL OF LINES WITH NaN: {removed_nan_lines} lines removed (from {lines_before_dropna} to {lines_after_dropna}).", Config.OUTPUT_TXT)
    else:
        save_to_txt(f"\nREMOVAL OF LINES WITH NaN: None of the lines contained NaN.", Config.OUTPUT_TXT)
    if df.empty:
        msg="DataFrame `df` is empty after dropna. Analysis cannot continue."
        print(msg)
        save_to_txt(f"\n{msg}", Config.OUTPUT_TXT)
        return
    save_to_txt(f"Dimensions after dropna: {df.shape}", Config.OUTPUT_TXT)
    buffer=StringIO()
    df.info(buf=buffer)
    info_str=buffer.getvalue()
    save_to_txt("\nInfo of the prepared DataFrame (AFTER dropna):\n" + info_str, Config.OUTPUT_TXT)

    if df[current_target_column_name_in_df].isna().sum() > 0:
        df[current_target_column_name_in_df]=df[current_target_column_name_in_df].fillna(0)
        save_to_txt(f"WARNING: NaNs in target '{current_target_column_name_in_df}' filled with 0.", Config.OUTPUT_TXT)
    unique_vals_target=df[current_target_column_name_in_df].unique()
    if not np.all(np.isin(unique_vals_target, [0, 1])):
        save_to_txt(f"WARNING: Target '{current_target_column_name_in_df}' with values != 0/1: {unique_vals_target}. Converting to 0/1.", Config.OUTPUT_TXT)
        df[current_target_column_name_in_df]=df[current_target_column_name_in_df].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)
    df[current_target_column_name_in_df]=df[current_target_column_name_in_df].astype(np.int8)
    save_to_txt(f"Target variable '{current_target_column_name_in_df}' processed to int8.", Config.OUTPUT_TXT)
    if not df.empty:
        save_to_txt(f"Target distribution '{current_target_column_name_in_df}' (BEFORE SMOTE):\n" + \
                    df[current_target_column_name_in_df].value_counts(normalize=True).round(4).to_string() +"\n" + \
                    df[current_target_column_name_in_df].value_counts().to_string(), Config.OUTPUT_TXT)
        
    # --- EXPLORATORY ANALYSIS ---
    save_to_txt("\n\n=== EXPLORATORY ANALYSIS (Based on `df` Processed) ===", Config.OUTPUT_TXT)
    numeric_cols_eda=df.select_dtypes(include=np.number).columns
    if len(numeric_cols_eda) > 1:
        save_to_txt("\n1. Univariate Analysis (Numerical):", Config.OUTPUT_TXT)
        save_to_txt(" - Generating histograms and boxplots...", Config.OUTPUT_TXT)
        for col_eda in numeric_cols_eda:
            if col_eda != current_target_column_name_in_df:
                try:
                    col_data=df[col_eda].dropna()

                    if col_data.empty:
                        save_to_txt(f" - WARNING: Column '{col_eda}' is empty after removing NaNs. Skipping graphs.", Config.OUTPUT_TXT)
                        continue

                    if col_data.nunique() <= 1:
                        save_to_txt(f" - WARNING: Column '{col_eda}' has only one unique value after removing NaNs. Skipping charts.", Config.OUTPUT_TXT)
                        continue

                    if not np.isfinite(col_data).all():
                        save_to_txt(f" - WARNING: Column '{col_eda}' contains non-finite values ​​(Inf/NaN). Skipping charts.", Config.OUTPUT_TXT)
                        continue

                    plt.figure(figsize=(5.5, 4.5))
                    plt.subplot(1, 2, 1)
                    sns.histplot(col_data, kde=True, bins=30)
                    # plt.title(f"Distribution {col_eda}")
                    plt.subplot(1, 2, 2)

                    sns.boxplot(x=col_data)

                    # plt.title(f"Boxplot {col_eda}")
                    plt.tight_layout()
                    plot_filename_eda=os.path.join(Config.OUTPUT_DIR, f'distribution_{col_eda}.png')
                    plt.savefig(plot_filename_eda, dpi=100)
                    plt.close()
                except Exception as e:
                    save_to_txt(f" - ERROR generating graph for '{col_eda}': {str(e)}", Config.OUTPUT_TXT)
                    plt.close()
        save_to_txt(" - Graphics saved in the output directory.", Config.OUTPUT_TXT)
    else:
        save_to_txt("\n1. Univariate Analysis (Numerical):\n - No numerical predictor variable to plot.", Config.OUTPUT_TXT)
    
    # --- PRELIMINARY MULTICOLINEARITY ANALYSIS ---
    save_to_txt("\n\n=== PRELIMINARY MULTICOLINEARITY ANALYSIS (NUMERICAL) ===", Config.OUTPUT_TXT)
    save_to_txt(f"Pair-by-pair analysis (Correlation > {Config.MAX_CORR}).", Config.OUTPUT_TXT)
    num_df_corr=df.select_dtypes(include=np.number).drop(columns=[current_target_column_name_in_df], errors='ignore')

    # Check if there are enough variables to calculate the correlation
    if not num_df_corr.empty and num_df_corr.shape[1] > 1:
        corr_matrix_val=num_df_corr.corr()

        mask_upper_triangle=np.triu(np.ones_like(corr_matrix_val, dtype=bool), k=1)

        stacked_corr_values=corr_matrix_val.abs().where(mask_upper_triangle).stack()

        high_corr_series=stacked_corr_values[stacked_corr_values > Config.MAX_CORR]

        high_corr_pair_val=[]
        if not high_corr_series.empty:
            for(idx1, idx2), corr_val in high_corr_series.items():
                high_corr_pair_val.append((idx1, idx2, corr_val))

        if high_corr_pair_val:
            save_to_txt(f"\nWARNING: {len(high_corr_pair_val)} NUMERIC pairs with correlation > {Config.MAX_CORR}:", Config.OUTPUT_TXT)
            for pair_val in sorted(high_corr_pair_val, key=lambda x_item: x_item[2], reverse=True):
                save_to_txt(f" - {pair_val[0]} x {pair_val[1]}: {pair_val[2]:.3f}", Config.OUTPUT_TXT)
        else:
            save_to_txt(f"\nNo high numerical correlation (> {Config.MAX_CORR}) detected.", Config.OUTPUT_TXT)

        try:
            plt.figure(figsize=(max(5.5, num_df_corr.shape[1] * 0.6), max(4.5, num_df_corr.shape[1] * 0.5)))
            mask_val=np.triu(np.ones_like(corr_matrix_val, dtype=bool))
            sns.heatmap(corr_matrix_val, mask=mask_val, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, annot_kws={"size": 7})
            # plt.title("Preliminary Numerical Correlation")
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            corr_plot_filename_val=os.path.join(Config.OUTPUT_DIR, "preliminary_numerical_correlation.png")
            plt.savefig(corr_plot_filename_val, dpi=300)
            plt.close()
            save_to_txt(f"\nSaved numerical correlation heatmap.", Config.OUTPUT_TXT)
        except Exception as e:
            save_to_txt(f"\nERROR generating correlation heatmap: {e}")
            plt.close()
    else:
        save_to_txt("\nInsufficient numeric variables to calculate correlation or empty DataFrame.", Config.OUTPUT_TXT)
    save_to_txt("NOTE: Full VIF will be created after dummies and corrections.", Config.OUTPUT_TXT)

    # --- DATA AND DUMMIES PREPARATION ---
    save_to_txt("\n\n=== DATA AND DUMMY PREPARATION (IF APPLICABLE) ===", Config.OUTPUT_TXT)
    if current_target_column_name_in_df not in df.columns:
        print(f"INTERNAL ERROR: Target '{current_target_column_name_in_df}' missed before dummies.")
        return
    y=df[current_target_column_name_in_df].copy()
    X_pre_dummies=df.drop(current_target_column_name_in_df, axis=1)

    categorical_cols_dummies=X_pre_dummies.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols_dummies) > 0:
        save_to_txt(f"\nConverting {len(categorical_cols_dummies)} categorical columns into dummies...", Config.OUTPUT_TXT)
        cols_to_dummy=[col for col in categorical_cols_dummies if X_pre_dummies[col].nunique() > 1]
        if cols_to_dummy:
            X=pd.get_dummies(X_pre_dummies, columns=cols_to_dummy, drop_first=True, dtype=np.int8)
            save_to_txt("Converted Categorical Columns: " + ", ".join(cols_to_dummy), Config.OUTPUT_TXT)
            numeric_original_cols=X_pre_dummies.select_dtypes(include=np.number).columns
            for num_col in numeric_original_cols:
                if num_col not in X.columns:
                    X[num_col]=X_pre_dummies[num_col]
        else:
            save_to_txt("No categorical column with more than one unique value to convert to dummies.", Config.OUTPUT_TXT)
            X=X_pre_dummies.copy()
    else:
        save_to_txt("\nNo categorical column found to convert to dummies.", Config.OUTPUT_TXT)
        X=X_pre_dummies.copy()

    # Ensures that X contains only numeric types after conversion and addition
    X=X.select_dtypes(include=np.number)
    save_to_txt(f"Number of predictors (numeric + dummies) before the VIF: {X.shape[1]}", Config.OUTPUT_TXT)
    if X.empty:
        msg="CRITICAL ERROR: DataFrame X empty after dummies."
        print(msg)
        save_to_txt(msg, Config.OUTPUT_TXT)
        return
    
    # --- VIF ANALYSIS ---
    X_post_vif1, final_vars_post_vif1, _=run_vif_analysis(X.copy(), Config.VIF_THRESHOLD, section_title="VIF Analysis - Stage 1 (Post-Dummies)")
    if X_post_vif1.empty or not final_vars_post_vif1:
        save_to_txt("\nNo variables remaining after VIF Step 1. Modeling cannot proceed.", Config.OUTPUT_TXT)
        X_model=pd.DataFrame()
    else:
        X_model=X_post_vif1.copy()

    # --- APPLICATION OF SMOTE ---
    if Config.APPLY_SMOTE and not X_model.empty and not y.empty:
        save_to_txt("\n\n=== SMOTE APPLICATION ===", Config.OUTPUT_TXT)
        value_counts_y_before_smote=y.value_counts()

        # Checks for significant imbalance
        # Considers it unbalanced if the smallest class is less than 49% of the total
        is_imbalanced =len(value_counts_y_before_smote) > 1 and (value_counts_y_before_smote.min() / value_counts_y_before_smote.sum()) < 0.49

        if is_imbalanced:
            save_to_txt(f"Imbalance detected. Applying SMOTE (k_neigh={Config.SMOTE_K_NEIGHBORS}, strat='{Config.SMOTE_SAMPLING_STRATEGY}')...", Config.OUTPUT_TXT)
            try:
                k_n_smote=Config.SMOTE_K_NEIGHBORS
                # Adjust k neighbors if the minority class is too small
                if value_counts_y_before_smote.min() <= k_n_smote:
                    k_n_smote=max(1, value_counts_y_before_smote.min() - 1)
                    if k_n_smote == 0 and value_counts_y_before_smote.min() == 1:
                        save_to_txt(f"SMOTE WARNING: Minority class has only 1 sample. SMOTE not feasible. Continuing without SMOTE.", Config.OUTPUT_TXT)
                        Config.APPLY_SMOTE=False
                        X_model=X_model
                        y=y
                    elif k_n_smote == 0:
                        save_to_txt(f"ERROR SMOTE: k_neighbors resulted in 0. Small minority class. Continuing with original data.", Config.OUTPUT_TXT)
                    else:
                        save_to_txt(f" Setting k_neighbors for SMOTE to: {k_n_smote} (minority class: {value_counts_y_before_smote.min()}).", Config.OUTPUT_TXT)
                        smote=SMOTE(sampling_strategy=Config.SMOTE_SAMPLING_STRATEGY, random_state=Config.SMOTE_RANDOM_STATE, k_neighbors=k_n_smote)
                        X_cols_b4_smote=X_model.columns.tolist()
                        y_name_b4_smote=y.name if hasattr(y, 'name') else current_target_column_name_in_df
                        X_smote_np, y_smote_np=smote.fit_resample(X_model, y)
                        save_to_txt(f"X dimensions before SMOTE: {X_model.shape}, y before SMOTE: {y.shape}", Config.OUTPUT_TXT)
                        X_model=pd.DataFrame(X_smote_np, columns=X_cols_b4_smote)
                        y=pd.Series(y_smote_np, name=y_name_b4_smote)
                        save_to_txt(f"Dimensions X after SMOTE: {X_model.shape}, y after SMOTE: {y.shape}", Config.OUTPUT_TXT)
                        save_to_txt("Target distribution AFTER SMOTE:\n" + y.value_counts(normalize=True).round(4).to_string() + "\n" + y.value_counts().to_string(), Config.OUTPUT_TXT)
                else:
                    smote=SMOTE(sampling_strategy=Config.SMOTE_SAMPLING_STRATEGY, random_state=Config.SMOTE_RANDOM_STATE, k_neighbors=k_n_smote)
                    X_cols_b4_smote=X_model.columns.tolist()
                    y_name_b4_smote=y.name if hasattr(y, 'name') else current_target_column_name_in_df
                    X_smote_np, y_smote_np=smote.fit_resample(X_model, y)
                    save_to_txt(f"X dimensions before SMOTE: {X_model.shape}, y before SMOTE: {y.shape}", Config.OUTPUT_TXT) 
                    X_model=pd.DataFrame(X_smote_np, columns=X_cols_b4_smote)
                    y=pd.Series(y_smote_np, name=y_name_b4_smote)
                    save_to_txt(f"Dimensions X after SMOTE: {X_model.shape}, y after SMOTE: {y.shape}", Config.OUTPUT_TXT)
                    save_to_txt("Target distribution AFTER SMOTE:\n" + y.value_counts(normalize=True).round(4).to_string() + "\n" + y.value_counts().to_string(), Config.OUTPUT_TXT)
            except ValueError as ve_smote:
                save_to_txt(f"SMOTE ERROR (ValueError): {ve_smote}. Check k_neighbors. Use original data.", Config.OUTPUT_TXT)
            except Exception as e_smote:
                save_to_txt(f"GENERAL ERROR SMOTE: {e_smote}. Using original data.\n{traceback.format_exc()}", Config.OUTPUT_TXT)
        else:
            if len(value_counts_y_before_smote) <= 1:
                save_to_txt("\nSMOTE not applied: Only one class present in the target.", Config.OUTPUT_TXT)
            else:
                save_to_txt("\nSMOTE not applied: Classes already considered relatively balanced.", Config.OUTPUT_TXT)
    elif not Config.APPLY_SMOTE:
        save_to_txt("\nSMOTE disabled by configuration.", Config.OUTPUT_TXT)
    elif X_model.empty:
        save_to_txt("\nSMOTE not applied: X_model empty.", Config.OUTPUT_TXT)
    elif y.empty:
        save_to_txt("\nSMOTE not applied: y empty.", Config.OUTPUT_TXT)
    # --- STEP: LASSO VARIABLE SELECTION (PRE-FILTERING) ---
    save_to_txt("\n\n=== STAGE: LASSO REGULARIZATION (L1 SELECTION) ===", Config.OUTPUT_TXT)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if X_model.empty or X_model.shape[1] == 0:
        save_to_txt("X_model empty. Skipping LASSO stage.", Config.OUTPUT_TXT)
    else:
        save_to_txt(f"Initial variables for LASSO: {X_model.columns.tolist()}", Config.OUTPUT_TXT)
        
        # 1. Padronização (Obrigatória para o LASSO funcionar corretamente)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_model)
        
        # 2. Configuração do LASSO
        # C=1.0 é o equilíbrio. Valores menores (ex: 0.1) são mais agressivos na remoção.
        lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
        lasso.fit(X_scaled, y)
        
        # 3. Análise de Coeficientes
        coefs = lasso.coef_[0]
        vars_kept_by_lasso = [X_model.columns[i] for i, c in enumerate(coefs) if c != 0]
        vars_removed_by_lasso = [X_model.columns[i] for i, c in enumerate(coefs) if c == 0]
        
        # 4. Registro do Passo a Passo no TXT
        save_to_txt(f"LASSO Penalty Analysis:", Config.OUTPUT_TXT)
        for i, var in enumerate(X_model.columns):
            status = "KEPT" if coefs[i] != 0 else "REMOVED (Zeroed)"
            save_to_txt(f" - {var}: Coef = {coefs[i]:.6f} [{status}]", Config.OUTPUT_TXT)
        
        save_to_txt(f"\nSummary of LASSO stage:", Config.OUTPUT_TXT)
        save_to_txt(f"Total variables removed: {len(vars_removed_by_lasso)}", Config.OUTPUT_TXT)
        save_to_txt(f"Total variables kept: {len(vars_kept_by_lasso)}", Config.OUTPUT_TXT)

        # 5. Atualização do X_model para as próximas etapas (P-valor / IC)
        if vars_kept_by_lasso:
            X_model = X_model[vars_kept_by_lasso]
            save_to_txt("X_model updated with LASSO selections for next stages.", Config.OUTPUT_TXT)
        else:
            save_to_txt("WARNING: LASSO removed all variables. Reverting to original X_model to avoid crash.", Config.OUTPUT_TXT)


    # --- LOGISTICS MODELING AND P-VALUE SELECTION ---
    save_to_txt("\n\n=== LOGISTICS MODELING AND FINAL SELECTION (Backward p-value) ===", Config.OUTPUT_TXT)
    final_model_results=None
    final_vars_selected_model=[]
    if X_model.empty or X_model.shape[1] == 0:
        save_to_txt("X_model empty or without columns. Skipping p-value modeling.", Config.OUTPUT_TXT)
    else:
        current_vars_selection_pval=X_model.columns.tolist()
        if not current_vars_selection_pval:
            save_to_txt("No column in X_model to start p-value selection.", Config.OUTPUT_TXT)
        else:
            save_to_txt(f"Starting selection by p-value with {len(current_vars_selection_pval)} vars. Alpha: {Config.ALPHA}", Config.OUTPUT_TXT)
            removed_vars_pval_list=[]
            iter_pval_count=0
            max_iter_pval_limit=len(current_vars_selection_pval) + 25
            while True:
                iter_pval_count += 1
                if iter_pval_count > max_iter_pval_limit:
                    save_to_txt(f"WARNING: Iteration limit ({max_iter_pval_limit}) reached in p-value selection.", Config.OUTPUT_TXT)
                    break
                if not current_vars_selection_pval:
                    save_to_txt("No variables remaining for modeling in p-value selection.", Config.OUTPUT_TXT)
                    break

                X_const_pval=sm.add_constant(X_model[current_vars_selection_pval], has_constant='add')
                try:
                    logit_model_pval_fit =sm.Logit(y, X_const_pval).fit(method='newton', maxiter=Config.MAX_ITERATION, disp=0)
                    if not logit_model_pval_fit.mle_retvals['converged']:
                        logit_model_pval_fit=sm.Logit(y,X_const_pval).fit(method='bfgs', maxiter=Config.MAX_ITERATION, disp=0)
                        if not logit_model_pval_fit.mle_retvals['converged']:
                            save_to_txt(f" - WARNING (Iter {iter_pval_count}): Model did not converge.", Config.OUTPUT_TXT)
                            p_values_for_no_converge=logit_model_pval_fit.pvalues.drop('const', errors='ignore') if hasattr(logit_model_pval_fit, 'pvalues') else pd.Series()
                            var_to_remove=p_values_for_no_converge.idxmax() if not p_values_for_no_converge else (current_vars_selection_pval[-1] if current_vars_selection_pval else None)
                            if var_to_remove and var_to_remove in current_vars_selection_pval:
                                current_vars_selection_pval.remove(var_to_remove)
                                removed_vars_pval_list.append((var_to_remove, 'No Convergence'))
                                save_to_txt(f" Removing '{var_to_remove}' due to non-convergence.", Config.OUTPUT_TXT)
                                continue
                            else:
                                break
                    p_values_iter=logit_model_pval_fit.pvalues.drop('const', errors='ignore')
                    if p_values_iter.empty:
                        final_model_results=logit_model_pval_fit
                        break
                    p_values_gt_alpha=p_values_iter[p_values_iter > Config.ALPHA]
                    if p_values_gt_alpha.empty:
                        final_model_results=logit_model_pval_fit
                        save_to_txt(f"\nP-value selection completed (Iter {iter_pval_count}). All {len(p_values_iter)} vars sig.", Config.OUTPUT_TXT)
                        break
                    else:
                        var_to_remove_pval=p_values_gt_alpha.idxmax()
                        max_p_val=p_values_gt_alpha.max()
                        if var_to_remove_pval in current_vars_selection_pval:
                            current_vars_selection_pval.remove(var_to_remove_pval)
                        removed_vars_pval_list.append((var_to_remove_pval, max_p_val))
                except (np.linalg.LinAlgError, ValueError) as e_linpval:
                    save_to_txt(f"ERRO Linalg/Val durante seleção p-valor (Iter {iter_pval_count}): {e_linpval}", Config.OUTPUT_TXT)
                    if current_vars_selection_pval:
                        var_rem_err=current_vars_selection_pval.pop()
                        removed_vars_pval_list.append((var_rem_err, f"ERROR ({type(e_linpval).__name__})"))
                        save_to_txt(f" Trying to remove '{var_rem_err}'.", Config.OUTPUT_TXT)
                        continue
                    else:
                        save_to_txt("Interrupting p-value selection.", Config.OUTPUT_TXT)
                        break
                except Exception as e_fatal_pval_sel:
                    save_to_txt(f"Fatal error p-value selection: {e_fatal_pval_sel}\n{traceback.format_exc()}", Config.OUTPUT_TXT)
                    break
            if removed_vars_pval_list:
                save_to_txt(f"\nVars removed (p-value > {Config.ALPHA} or error):", Config.OUTPUT_TXT)
                for var_r, p_r in removed_vars_pval_list:
                    save_to_txt(f"- {var_r}: {p_r if isinstance(p_r, str) else f'p={p_r:.4f}'}", Config.OUTPUT_TXT)
            if final_model_results:
                final_vars_selected_model=final_model_results.pvalues.drop('const', errors='ignore').index.tolist()
                save_to_txt(f"\nFINAL Vars (POST P-VALUE): {len(final_vars_selected_model)} {final_vars_selected_model if final_vars_selected_model else 'None'}", Config.OUTPUT_TXT)
                if final_vars_selected_model:
                    save_to_txt("Summary (POST P-VALUE):\n" + final_model_results.summary2().as_text(), Config.OUTPUT_TXT)
            else:
                save_to_txt("\nNo final model after p-value.", Config.OUTPUT_TXT)
                final_vars_selected_model=[]

    # Removal by OR IC

    model_after_pval_selection=final_model_results
    vars_after_pval_selection=list(final_vars_selected_model)

    save_to_txt("\n\n=== ADDITIONAL VERIFICATION/REMOVAL BY CONFIDENCE INTERVAL OF THE OR (IC OR vs 1.0) ===", Config.OUTPUT_TXT)

    if vars_after_pval_selection and not X_model.empty and not y.empty:
        save_to_txt(f"Starting verification of {len(vars_after_pval_selection)} vars of the p-value selection: {', '.join(vars_after_pval_selection)}", Config.OUTPUT_TXT)
        model_to_use_for_ic_details=None
        if model_after_pval_selection and hasattr(model_after_pval_selection, 'mle_retvals') and model_after_pval_selection.mle_retvals.get('converged', False):
            model_to_use_for_ic_details=model_after_pval_selection
        elif vars_after_pval_selection:
            save_to_txt("Model p-value not available/converged. Adjusting for IC verification...", Config.OUTPUT_TXT)
            X_temp_ic_check=sm.add_constant(X_model[vars_after_pval_selection], has_constant='add')
            try:
                temp_model_ic_fit=sm.Logit(y, X_temp_ic_check).fit(method='newton', maxiter=Config.MAX_ITERATION, disp=0)
                if not temp_model_ic_fit.mle_retvals['converged']:
                    temp_model_ic_fit=sm.Logit(y, X_temp_ic_check).fit(method='bfgs', maxiter=Config.MAX_ITERATION, disp=0)
                if temp_model_ic_fit.mle_retvals['converged']: 
                    model_to_use_for_ic_details=temp_model_ic_fit
                else:
                    save_to_txt("WARNING: Temporary IC model did not converge.", Config.OUTPUT_TXT)
            except Exception as e_refit_ic_check:
                save_to_txt(f" ERROR reset for IC: {e_refit_ic_check}", Config.OUTPUT_TXT)

        if model_to_use_for_ic_details:
            conf_int_coeffs_ic_check=model_to_use_for_ic_details.conf_int()
            vars_to_remove_by_ic_check=[]
            vars_kept_after_ic_check_list=[]
            for var_name_for_ic in vars_after_pval_selection:
                try:
                    beta_ci_lower_ic, beta_ci_upper_ic=conf_int_coeffs_ic_check.loc[var_name_for_ic, 0], conf_int_coeffs_ic_check.loc[var_name_for_ic, 1]
                except KeyError:
                    save_to_txt(f" WARNING: Var '{var_name_for_ic}' not in ICs. Keeping.", Config.OUTPUT_TXT)
                    vars_kept_after_ic_check_list.append(var_name_for_ic)
                    continue
                or_ic_lower_precise_val=np.exp(beta_ci_lower_ic)
                or_ic_upper_precise_val=np.exp(beta_ci_upper_ic)
                msg_detailed_ic_log=generate_message_significance_ic_or(or_ic_lower_precise_val, or_ic_upper_precise_val, for_log_remotion=True)
                is_one_in_or_ic=False
                if not (pd.isna(or_ic_lower_precise_val) or pd.isna(or_ic_upper_precise_val)):
                    is_one_in_or_ic=(or_ic_lower_precise_val <= 1.0 <= or_ic_upper_precise_val)
                if is_one_in_or_ic:
                    vars_to_remove_by_ic_check.append(var_name_for_ic)
                    save_to_txt(f" - '{var_name_for_ic}': CANDIDATE FOR REMOVAL. {msg_detailed_ic_log}", Config.OUTPUT_TXT)
                else:
                    vars_kept_after_ic_check_list.append(var_name_for_ic)
                    save_to_txt(f" - '{var_name_for_ic}': KEPT. {msg_detailed_ic_log}", Config.OUTPUT_TXT)
            
            if vars_to_remove_by_ic_check:
                save_to_txt(f"\nVars REMOVED (IC OR contains 1.0): {', '.join(vars_to_remove_by_ic_check) if vars_to_remove_by_ic_check else 'None'}", Config.OUTPUT_TXT)
                final_vars_selected_model=list(vars_kept_after_ic_check_list)
                if final_vars_selected_model:
                    save_to_txt(f"Refitting final model with {len(final_vars_selected_model)} vars: {', '.join(final_vars_selected_model)}", Config.OUTPUT_TXT)
                    X_truly_final_model_fit=sm.add_constant(X_model[final_vars_selected_model], has_constant='add')
                    try:
                        final_model_results=sm.Logit(y, X_truly_final_model_fit).fit(method='newton', maxiter=Config.MAX_ITERATION, disp=0)
                        if not final_model_results.mle_retvals['converged']:
                            final_model_results=sm.Logit(y, X_truly_final_model_fit).fit(method='bfgs', maxiter=Config.MAX_ITERATION, disp=0)
                        if final_model_results.mle_retvals['converged']:
                            save_to_txt("Final model (post-IC) adjusted.\n--- Summary (POST-IC AND ADJUSTMENT) ---\n" + final_model_results.summary2().as_text(), Config.OUTPUT_TXT)
                        else:
                            save_to_txt("WARNING: Final model (post-IC) did NOT converge.", Config.OUTPUT_TXT)
                            final_model_results=None
                    except Exception as e_final_refit_exc:
                        save_to_txt(f"ERROR final readjustment post-IC: {e_final_refit_exc}", Config.OUTPUT_TXT)
                        final_model_results=None
                else:
                    save_to_txt("No variables remaining after removal by IC.", Config.OUTPUT_TXT)
                    final_model_results=None
            else:
                save_to_txt("\nNo additional variables removed by IC.", Config.OUTPUT_TXT)
                final_model_results=model_to_use_for_ic_details
        else:
            save_to_txt("Unable to obtain a model for IC verification. Skipping further removal.", Config.OUTPUT_TXT)
            final_model_results=model_after_pval_selection
            final_vars_selected_model=list(vars_after_pval_selection)
    elif X_model.empty or y.empty:
        save_to_txt("X_model or y empty. Skipping IC check.", Config.OUTPUT_TXT)
    else:
        save_to_txt("No p-value selection variable to check with IC.", Config.OUTPUT_TXT)
    save_to_txt(f"\nFINAL Vars (AFTER ALL SELECTIONS): {len(final_vars_selected_model) if final_vars_selected_model else '0'} {final_vars_selected_model if final_vars_selected_model else 'None'}", Config.OUTPUT_TXT)

    # --- INTERPRETATION OF ODDS RATIOS (FROM THE TRULY FINAL MODEL) ---
    if final_model_results and final_vars_selected_model:
        save_to_txt("\n\n=== INTERPRETATION (ODDS RATIOS OF THE FINAL MODEL AFTER TOTAL SELECTION) ===", Config.OUTPUT_TXT)
        try:
            params_or=final_model_results.params.drop('const', errors='ignore')
            if not params_or.empty:
                conf_int_or_aligned=final_model_results.conf_int().reindex(params_or.index)
                p_values_aligned=final_model_results.pvalues.reindex(params_or.index)
                odds_ratios_df=pd.DataFrame({
                    'Coef (β)': params_or, 'Odds Ratio': np.exp(params_or),
                    'IC 95% Inf': np.exp(conf_int_or_aligned.iloc[:, 0]),
                    'IC 95% Sup': np.exp(conf_int_or_aligned.iloc[:, 1]),
                    'p-valor': p_values_aligned})
                significance_notes_or=[]
                for _, row_or_ft in odds_ratios_df.iterrows():
                    msg_ft=generate_message_significance_ic_or(row_or_ft['IC 95% Inf'], row_or_ft['IC 95% Sup'], for_log_remotion=False)
                    significance_notes_or.append(msg_ft)
                odds_ratios_df['Significance (IC OR vs 1.0)']=significance_notes_or
                save_to_txt("\n--- Significance Analysis using the CI of the OR (CI OR vs 1.0) ---", Config.OUTPUT_TXT)
                save_to_txt("This column evaluates whether the OR of each variable is statistically different from 1.0.", Config.OUTPUT_TXT)
                save_to_txt("- 'Significant (IC of OR does NOT contain 1.0)': IC does NOT include 1.0. Significant effect (p < ~0.05).", Config.OUTPUT_TXT)
                save_to_txt("- 'Significant (IC does NOT contain 1.0; limits rounded to 1.0000)': Even rounded to 1.0000, the precise IC does NOT include 1.0. Effect sig.", Config.OUTPUT_TXT)
                save_to_txt("- 'Not Significant (CI of OR contains 1.0)': CI INCLUDES 1.0. No evidence of effect (p >= ~0.05).", Config.OUTPUT_TXT)
                save_to_txt("- 'IC Unavailable': It was not possible to calculate the IC.", Config.OUTPUT_TXT)
                save_to_txt(odds_ratios_df.round(4).to_string(), Config.OUTPUT_TXT)
            else:
                save_to_txt("No predictor coefficient in the final model for the OR table.", Config.OUTPUT_TXT)
        except Exception as e_or_final_exc_2:
            save_to_txt(f"ERROR in final OR table: {e_or_final_exc_2}\n{traceback.format_exc()}", Config.OUTPUT_TXT)
    elif not final_vars_selected_model and final_model_results:
        save_to_txt("\n\n=== INTERPRETATION (ODDS RATIOS FINAL) ===\nFinal model contains only the intercept.", Config.OUTPUT_TXT)
    else:
        save_to_txt("\n\n=== INTERPRETATION (FINAL ODDS RATIOS) ===\no final model to calculate Odds Ratios.", Config.OUTPUT_TXT)

    # --- INFLUENCE OF VARIABLES ---
    if final_model_results and final_vars_selected_model and not X_model.empty:
        save_to_txt("\n\n=== INFLUENCE OF VARIABLES (Standardized Coefficients OF THE FINAL MODEL) ===", Config.OUTPUT_TXT)
        try:
            coeffs_inf_final=final_model_results.params.drop('const', errors='ignore')
            if not coeffs_inf_final.empty:
                vars_for_std_final=coeffs_inf_final.index.intersection(X_model.columns)
                if not vars_for_std_final.empty:
                    stdevs_inf_final=X_model[vars_for_std_final].std(skipna=True).replace(0, 1e-9)
                    coeffs_align_final, stdevs_align_final=coeffs_inf_final.align(stdevs_inf_final, join='inner')
                    if not coeffs_align_final.empty:
                        standardized_coeffs_final=coeffs_align_final * stdevs_align_final
                        influence_df_final=pd.DataFrame({'Var': standardized_coeffs_final.index, 'Standardized Coef': standardized_coeffs_final.values})
                        influence_df_final['Influence (Abs)']=np.abs(influence_df_final['Standardized Coef'])
                        influence_df_final=influence_df_final.sort_values(by='Influence (Abs)', ascending=False).reset_index(drop=True)
                        save_to_txt("Ranking: abs(Coefficient * Predictor Standard Deviation)", Config.OUTPUT_TXT)
                        save_to_txt(influence_df_final[['Var', 'Standardized Coef']].round(4).to_string(index=True), Config.OUTPUT_TXT)
                    else:
                        save_to_txt("No common variables after alignment for influence calculation.", Config.OUTPUT_TXT)
                else:
                    save_to_txt("Model variables not found in X_model for standard calculation.", Config.OUTPUT_TXT)
            else:
                save_to_txt("No predictor variable in the final model to calculate influence.", Config.OUTPUT_TXT)
        except Exception as e_inf_final_exc:
            save_to_txt(f"ERROR ranking influence: {e_inf_final_exc}\n{traceback.format_exc()}", Config.OUTPUT_TXT)
    elif not final_vars_selected_model and final_model_results:
        save_to_txt("\n\n=== INFLUENCE OF VARIABLES ===\nFinal model contains only the intercept.", Config.OUTPUT_TXT)
    else:
        save_to_txt("\n\n=== INFLUENCE OF VARIABLES ===\nNo final model or empty X_model to calculate influence.", Config.OUTPUT_TXT)

    # --- MODEL EVALUATION ---
    if final_model_results and not y.empty:
        if final_vars_selected_model and not X_model.empty and all(v_eval_f in X_model.columns for v_eval_f in final_vars_selected_model):
            save_to_txt("\n\n=== FINAL MODEL EVALUATION (POST-TOTAL SELECTION, IN THE TRAINING DATASET) ===", Config.OUTPUT_TXT)
            save_to_txt(f"Metrics calculated using threshold: {Config.PREDICTION_THRESHOLD}", Config.OUTPUT_TXT)
            X_eval_truly_final=sm.add_constant(X_model[final_vars_selected_model], has_constant='add')
            try:
                predicted_probs_truly_final=final_model_results.predict(X_eval_truly_final)
                predicted_labels_truly_final=(predicted_probs_truly_final >= Config.PREDICTION_THRESHOLD).astype(int)
                accuracy_tf=accuracy_score(y, predicted_labels_truly_final)
                precision_tf=precision_score(y, predicted_labels_truly_final, zero_division=0)
                recall_tf=recall_score(y, predicted_labels_truly_final, zero_division=0)
                f1_tf=f1_score(y, predicted_labels_truly_final, zero_division=0)
                cm_tf=confusion_matrix(y, predicted_labels_truly_final)
                class_report_tf=classification_report(y, predicted_labels_truly_final, zero_division=0, digits=4)
                roc_auc_tf=np.nan
                fpr_tf, tpr_tf=None, None
                if len(np.unique(y)) > 1:
                    try:
                        fpr_tf, tpr_tf, _=roc_curve(y,predicted_probs_truly_final)
                        roc_auc_tf=auc(fpr_tf, tpr_tf)
                    except ValueError as e_roc_calc_tf:
                        save_to_txt(f" WARNING (Final ROC/AUC calculation): {e_roc_calc_tf}", Config.OUTPUT_TXT)
                save_to_txt("\n--- Metrics ---", Config.OUTPUT_TXT)
                save_to_txt(f"Accuracy:{accuracy_tf:.4f}, Precision:{precision_tf:.4f}, Recall:{recall_tf:.4f}, F1:{f1_tf:.4f}, AUC:{roc_auc_tf:.4f}", Config.OUTPUT_TXT)
                save_to_txt("\nConfusion Matrix:\n" + str(cm_tf), Config.OUTPUT_TXT)
                save_to_txt("\nClassification Report:\n" + class_report_tf, Config.OUTPUT_TXT)
                if fpr_tf is not None and tpr_tf is not None and not pd.isna(roc_auc_tf):
                    try:
                        plt.figure(figsize=(5.5, 4.5))
                        plt.plot(fpr_tf, tpr_tf, color='darkorange', lw=2, label=f'ROC (AUC={roc_auc_tf:.3f})')
                        plt.plot([0,1],[0,1],color='navy',lw=1,linestyle='--')
                        plt.xlim([0,1]); plt.ylim([0,1.05])
                        plt.xlabel('FP Rate')
                        plt.ylabel('TP Rate')
                        # plt.title(f'Curva ROC Final ({filename_suffix.replace("_", " ").strip()})')
                        plt.legend(loc="lower right")
                        plt.grid(alpha=0.6)
                        plt.tight_layout()
                        roc_plot_filename_tf=os.path.join(Config.OUTPUT_DIR, f"roc_curve_TRULY_FINAL{filename_suffix}.png")
                        plt.savefig(roc_plot_filename_tf, dpi=100)
                        plt.close()
                        save_to_txt(f"\nFinal ROC graph saved.", Config.OUTPUT_TXT)
                    except Exception as e_roc_plot_tf:
                        save_to_txt(f"ERROR plot ROC final: {e_roc_plot_tf}", Config.OUTPUT_TXT)
                        plt.close()
                else:
                    save_to_txt(f"\nFinal ROC graph not generated.", Config.OUTPUT_TXT)
            except Exception as e_eval_tf_exc:
                save_to_txt(f"ERROR true final evaluation: {e_eval_tf_exc}\n{traceback.format_exc()}", Config.OUTPUT_TXT)
        elif not final_vars_selected_model:
            save_to_txt("\n\n=== FINAL MODEL EVALUATION ===\nFinal model contains no predictors. Limited evaluation.", Config.OUTPUT_TXT)
            if not y.empty:
                try:
                    majority_class=y.mode()[0]
                    baseline_accuracy=(y == majority_class).mean()
                    save_to_txt(f" Baseline accuracy: {baseline_accuracy:.4f}", Config.OUTPUT_TXT)
                except Exception:
                    pass
    else:
        save_to_txt("\n\n=== FINAL MODEL EVALUATION ===\nNo final model for evaluation or y is empty.", Config.OUTPUT_TXT)
    
    # --- GENERATE FINAL TABLE WITH SELECTED COLUMNS ---
    save_to_txt("\n\n=== GENERATING THE FINAL TABLE WITH SELECTED COLUMNS ===", Config.OUTPUT_TXT)
    if final_vars_selected_model:
        final_columns_for_table=[current_target_column_name_in_df] + final_vars_selected_model

        df_original_clean_names=dataframe_input.copy()
        df_original_clean_names.columns=[re.sub(r'[^a-z0-9_]+', '', col.strip().lower().replace('.', '_')) for col in df_original_clean_names.columns]
        existent_columns=[col for col in final_columns_for_table if col in df_original_clean_names.columns]

        if len(existent_columns) == len(final_columns_for_table):
            filtered_final_table=df_original_clean_names[existent_columns]
            output_table_file=os.path.join(Config.OUTPUT_DIR, f'table_selected_columns{filename_suffix}.txt')

            try:
                filtered_final_table.to_csv(output_table_file, sep='\t', index=False, encoding='utf-8')
                msg_success=f"Table with selected columns successfully saved to: '{output_table_file}'"
                print(msg_success)
                save_to_txt(msg_success, Config.OUTPUT_TXT)
                save_to_txt(f"The table contains {filtered_final_table.shape[0]} rows and {filtered_final_table.shape[1]} columns.", Config.OUTPUT_TXT)
            except Exception as e_save_table:
                error_msg=f"ERROR saving the final table of selected columns: {e_save_table}"
                print(error_msg)
                save_to_txt(error_msg, Config.OUTPUT_TXT)
        else:
            missing_columns=set(final_columns_for_table) - set(existent_columns)
            msg_warning=f"WARNING: The following selected columns were not found in the original DataFrame: {missing_columns}. The table was not generated."
            print(msg_warning)
            save_to_txt(msg_warning, Config.OUTPUT_TXT)
    else:
        msg="No variables were selected in the final model. No output table with data was generated."
        print(msg)
        save_to_txt(msg, Config.OUTPUT_TXT)

    script_end_time=pd.Timestamp.now()
    save_to_txt(f'\n\nEND OF REPORT ({filename_suffix.replace("_", " ").strip().upper()}) ({script_end_time.strftime("%Y-%m-%d %H:%M:%S")})', Config.OUTPUT_TXT)
    save_to_txt(f'Runtime: {script_end_time - script_start_time}', Config.OUTPUT_TXT)
    print(f"\nAnalysis ({filename_suffix.replace('_', ' ').strip().upper()}) completed. Results in '{Config.OUTPUT_TXT}'.")

# %%
if __name__ == "__main__":
    if final_df.empty:
        print("\nDataFrame `final_df` is empty or not loaded. Logistic analysis cannot continue.")
    else:
        print("\n--- STARTING ANALYSIS BLOCK ---")
        execute_logistics_analysis(dataframe_input=final_df.copy(), apply_smote=False, filename_suffix='_without_SMOTE_LASSO_9')
        execute_logistics_analysis(dataframe_input=final_df.copy(), apply_smote=True, filename_suffix='_with_SMOTE_LASSO_9')
    print("\n--- ANALYSIS SCRIPT COMPLETED ---")


