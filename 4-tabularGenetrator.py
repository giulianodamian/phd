# %%
import glob
from astropy.io import fits
import os
import traceback
import pandas as pd
import numpy as np
from astropy.table import Table
# Combines data from FITS and TXT files (from steps 2 and 3) based on coordinates, enriches with data from an external file (DRPALL)

class ConfigA:
    DIRECT_IN = "/data/public/sdss/manga/mcubes/*MEGACUBE.fits"
    TXT_DIR = "/scratch/users/giuliano.damian/text/2_agn_output_lists"
    DRPALL_FILE = "/scratch/users/giuliano.damian/data/other_data/manga_data_drpall_extracted.txt"
    TXT_SUFFIX = "-EMISSION_AGNxNonAGN_list.txt"
    FITS_SUFFIX = "-MEGACUBE.fits"
    OUTPUT_TXT_FILE = "/scratch/users/giuliano.damian/text/4-df_code.txt"
    MERGE_COLUMN_FINAL_DF = 'source_file'
    DRPALL_KEY_COLUMN_NAME_IN_HEADER = 'plateifu'

# %%
def load_and_prepare_drpall_data(filepath, configured_key_column_name):

    print(f"\nReading and preparing DRPALL file: {filepath} ---")

    try:
        header_line_content = None

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    header_line_content = stripped_line
                    break

        if header_line_content is None:
            print("ERROR: Header not found.")
            return pd.DataFrame()

        processed_column_names = header_line_content.lstrip('#').strip().split()

        print(f"Column names detected: {processed_column_names}")

        # ✅ DEFINA AQUI AS COLUNAS QUE VOCÊ QUER
        colunas_desejadas = [
            configured_key_column_name,
            'nsa_sersic_mass',
            'nsa_sersic_ba',
            'nsa_sersic_n',
            'nsa_extinction_r',
            'nsa_sersic_absmag_r'

        ]

        df_drpall = pd.read_csv(
            filepath,
            sep=r'\s+',
            header=None,
            names=processed_column_names,
            usecols=colunas_desejadas,   # 👈 agora existe
            skiprows=1,
            comment="#",
            dtype=str
        )

        if df_drpall.empty:
            print("DRPALL vazio.")
            return pd.DataFrame()

        df_drpall = df_drpall.rename(
            columns={configured_key_column_name: '_drpall_merge_key_'}
        )

        df_drpall['_drpall_merge_key_'] = (
            df_drpall['_drpall_merge_key_']
            .astype(str)
            .str.strip()
        )

        # Converter apenas colunas numéricas
        for col in ['nsa_sersic_mass', 'nsa_sersic_ba']:
            if col in df_drpall.columns:
                df_drpall[col] = pd.to_numeric(df_drpall[col], errors='coerce')

        print(f"DRPALL pronto com {df_drpall.shape[1]} colunas.")

        return df_drpall

    except Exception as e:
        print(f"Erro: {e}")
        traceback.print_exc()
        return pd.DataFrame()

# %%
print('\n--- STARTING GENERATION OF `final_df` DATAFRAME (FITS/TXT) ---')

input_fits_pattern = ConfigA.DIRECT_IN
fits_list = glob.glob(input_fits_pattern)

final_df = pd.DataFrame()
all_combined_data = []

if not fits_list:
    print(f"No FITS files found matching pattern: {input_fits_pattern}")
else:
    total_pixels_processed = 0
    total_valid_matches = 0
    total_ignored_matches = 0
    for fits_path in fits_list:
        fits_filename = os.path.basename(fits_path)
        if not fits_filename.endswith(ConfigA.FITS_SUFFIX):
            continue
        base_name = fits_filename[:-len(ConfigA.FITS_SUFFIX)]
        txt_filename = base_name + ConfigA.TXT_SUFFIX
        txt_path = os.path.join(ConfigA.TXT_DIR, txt_filename)
        if not os.path.exists(txt_path):
            continue
        try:
            txt_lookup_pair = {}    
            with open(txt_path, 'r') as f_txt:
                for line in f_txt:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 3:
                            try: 
                                y_txt, x_txt, txt_value = int(parts[0]), int(parts[1]), float(parts[2])
                            except ValueError: 
                                continue
                            txt_lookup_pair[(x_txt, y_txt)] = txt_value
            
            with fits.open(fits_path) as hdul:
                if len(hdul) <= 25:
                    continue
                hdu_popBins = hdul[25].data
                if hdu_popBins is None or not isinstance(hdu_popBins, np.ndarray) or hdu_popBins.ndim != 3:
                    continue
                n_params, n_y, n_x = hdu_popBins.shape
                total_pixels_processed += (n_y * n_x)
                for y_coord_fits in range(n_y):
                    for x_coord_fits in range(n_x):
                        current_coodinate = (x_coord_fits, y_coord_fits)
                        if current_coodinate in txt_lookup_pair:
                            corresponding_txt_value = txt_lookup_pair[current_coodinate]
                            if corresponding_txt_value == -1:
                                total_ignored_matches += 1
                                continue
                            total_valid_matches += 1
                            all_combined_data.append({
                                ConfigA.MERGE_COLUMN_FINAL_DF: base_name, 'x': x_coord_fits, 'y': y_coord_fits,
                                'valor_txt': corresponding_txt_value,
                                'xyy_fits': hdu_popBins[1,y_coord_fits,x_coord_fits], 'xy0_fits': hdu_popBins[2,y_coord_fits,x_coord_fits],
                                'xiy_fits': hdu_popBins[3,y_coord_fits,x_coord_fits], 'xii_fits': hdu_popBins[4,y_coord_fits,x_coord_fits],
                                'xio_fits': hdu_popBins[5,y_coord_fits,x_coord_fits], 'xo_fits': hdu_popBins[6,y_coord_fits,x_coord_fits],
                                'sfr_30E6_fits': hdu_popBins[18,y_coord_fits,x_coord_fits], 'Av_fits': hdu_popBins[22,y_coord_fits,x_coord_fits],
                                'mage_L_fits': hdu_popBins[23,y_coord_fits,x_coord_fits], 'Mz_L_fits': hdu_popBins[25,y_coord_fits,x_coord_fits],
                                'sigma_star_fits': hdu_popBins[30,y_coord_fits,x_coord_fits] if n_params > 30 else np.nan
                            })
        except Exception as e: print(f"ERROR processing '{base_name}': {e}"); traceback.print_exc(); continue
    print(f"FITS/TXT processing completed. {len(all_combined_data)} base data records generated.")
    print(f"  Total FITS pixels processed: {total_pixels_processed}")
    print(f"  Total valid matches: {total_valid_matches}")
    print(f"  Total ignored matches (-1): {total_ignored_matches}")

# %%
if all_combined_data:
    print("\nCreating `final_df` DataFrame with combined FITS/TXT data...")
    final_df = pd.DataFrame(all_combined_data)
    print(f"`final_df` DataFrame (FITS/TXT) created with {len(final_df)} rows and {len(final_df.columns)} columns.")

    df_drpall_processed = load_and_prepare_drpall_data(
        ConfigA.DRPALL_FILE,
        ConfigA.DRPALL_KEY_COLUMN_NAME_IN_HEADER # Passes the configured key name
    )
    if not df_drpall_processed.empty and not final_df.empty:
        print(f"\n--- Performing MERGE with processed DRPALL data ---")
        original_final_df_columns = list(final_df.columns)
        if ConfigA.MERGE_COLUMN_FINAL_DF not in final_df.columns:
            print(f"CRITICAL ERROR: Merge column '{ConfigA.MERGE_COLUMN_FINAL_DF}' not found in final_df. Aborting merge.")
        else:
            original_final_df_columns = list(final_df.columns)
            print("Exemplo final_df:")
            print(final_df['source_file'].unique()[:5])

            print("\nExemplo DRPALL:")
            print(df_drpall_processed['_drpall_merge_key_'].unique()[:5])
            final_df[ConfigA.MERGE_COLUMN_FINAL_DF] = (
                final_df[ConfigA.MERGE_COLUMN_FINAL_DF]
                .astype(str)
                .str.replace("manga-", "", regex=False)
                .str.strip()
)
            original_final_df_columns = list(final_df.columns)
            original_final_df_columns_set = set(original_final_df_columns)

            final_df_merged = pd.merge(
                final_df, df_drpall_processed,
                left_on = ConfigA.MERGE_COLUMN_FINAL_DF, right_on = '_drpall_merge_key_',
                how = 'left'
            )
            if '_drpall_merge_key_' in final_df_merged.columns:
                final_df_merged.drop(columns = ['_drpall_merge_key_'], inplace = True)
            
            num_rows_before_merge, num_rows_after_merge = len(final_df), len(final_df_merged)
            print(f"Merge completed. Rows before: {num_rows_before_merge}, Rows after: {num_rows_after_merge}")
            if num_rows_after_merge > num_rows_before_merge:
                print("WARNING: Number of rows increased after merge (check for duplicates in DRPALL key).")
            added_columns = set(final_df_merged.columns) - original_final_df_columns_set
            if added_columns:
                print(f"Columns added from DRPALL: {', '.join(sorted(list(added_columns)))}")
            else:
                print("No new columns appear to have been added from DRPALL.")
            final_df = final_df_merged

            colunas_drpall_desejadas = ['nsa_sersic_mass', 'nsa_sersic_ba', 'nsa_sersic_n','nsa_extinction_r', 'nsa_sersic_absmag_r']

            final_df = final_df[original_final_df_columns + colunas_drpall_desejadas]


            print(f"`final_df` DataFrame (after merge) has {len(final_df)} rows and {len(final_df.columns)} columns.")
    elif final_df.empty:
        print("`final_df` DataFrame (FITS/TXT) is empty. Merge with DRPALL not performed.")
    elif df_drpall_processed.empty:
        print("DRPALL DataFrame is empty or not processed. Merge not performed.")
    
    if not final_df.empty:
        try:
            print(f"\n--- Saving final DataFrame to '{ConfigA.OUTPUT_TXT_FILE}' ---")
            final_df.to_csv(ConfigA.OUTPUT_TXT_FILE, sep='\t', index=False, encoding='utf-8', na_rep='NaN')
            print(f"DataFrame successfully saved to '{ConfigA.OUTPUT_TXT_FILE}'")
        except Exception as e: print(f"ERROR saving DataFrame: {e}"); traceback.print_exc()
    else: print("\nNo data to save. `final_df` is empty.")
else: print("\nNo FITS/TXT data found. `final_df` is empty.")


