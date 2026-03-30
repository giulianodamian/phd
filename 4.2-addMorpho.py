# %%
import numpy as np
from astropy.io import fits
import pandas as pd
import os

fits_file_path = '/scratch/users/giuliano.damian/data/other_data/manga_visual_morpho-2.0.1.fits'
txt_file_path = '/scratch/users/giuliano.damian/text/4.1-df_code.txt'
output_txt_file_path = '/scratch/users/giuliano.damian/text/4.2-df_code.txt'

print(f"--- Trying to read original TXT file: {txt_file_path} ---")

df_txt = None
if not os.path.exists(txt_file_path):
    print(f"ERROR: TXT file '{txt_file_path}' not found.")
else:
    try:
        df_txt = pd.read_csv(txt_file_path, sep = r'\s+', header = 0, engine = 'python')
        print("Original TXT file successfully read into a Pandas DataFrame!")
        print(f"Original TXT columns: {df_txt.columns.tolist()}")
        print(f"Number of rows in original TXT: {len(df_txt)}")
    except Exception as e:
        print(f"ERROR reading TXT file '{txt_file_path}': {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*50 + "\n")



# %%
print(f"--- Trying to read FITS file: {fits_file_path} ---")

df_fits_data = None
if not os.path.exists(fits_file_path):
    print(f"ERROR: FITS file '{fits_file_path}' not found.")

else:
    try:
        with fits.open(fits_file_path) as hdul:
            print("FITS file opened successfully!")

            joined_hdu_found = False

            for hdu in hdul:
                if hdu.name == 'Joined' and hdu.data is not None:
                    joined_hdu_found = True
                    print("\nHDU 'Joined' found! Extracting relevant data...")

                    columns_from_fits = ['plateifu', 'TType', 'Bars', 'Tidal']

                    fits_data_dict = {}
                    all_cols_present = True

                    for col_name in columns_from_fits:
                        if col_name in hdu.data.names:
                            if col_name == 'plateifu':
                                fits_data_dict['plateifu_key'] = [
                                    str(val).strip().lower().replace('manga-','')
                                    for val in hdu.data[col_name]]                            
                            else:
                                fits_data_dict[col_name] = hdu.data[col_name]
                        else:
                            print(f"WARNING: Column '{col_name}' not found in HDU 'Joined'.")
                            all_cols_present = False
                            fits_data_dict[col_name] = np.full(len(hdu.data), np.nan)

                    if all_cols_present:
                        df_fits_data = pd.DataFrame({
                            'plateifu_key': fits_data_dict['plateifu_key'],
                            'TType': fits_data_dict['TType'],
                            'Bars': fits_data_dict['Bars'],
                            'Tidal': fits_data_dict['Tidal']
                        })
                        print("FITS data extracted to auxiliary DataFrame!")
                        print(f"First 5 rows of FITS auxiliary DataFrame:\n{df_fits_data.head()}")
                        print(f"Number of rows in FITS auxiliary DataFrame: {len(df_fits_data)}")
                    else:
                        print("Could not extract all required columns from FITS.")
                    break

            if not joined_hdu_found:
                print("WARNING: HDU 'Joined' was not found in this FITS file.")

    except Exception as e:
        print(f"ERROR reading FITS file '{fits_file_path}': {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*50 + "\n")

# %%
print("--- Starting value mapping and assignment ---")

if df_txt is not None and df_fits_data is not None:
    try:
        ttype_map = dict(zip(df_fits_data['plateifu_key'], df_fits_data['TType']))
        bars_map = dict(zip(df_fits_data['plateifu_key'], df_fits_data['Bars']))
        tidal_map = dict(zip(df_fits_data['plateifu_key'], df_fits_data['Tidal']))

        df_txt['TType'] = np.nan
        df_txt['Bars'] = np.nan
        df_txt['Tidal'] = np.nan
        
        df_txt['TType'] = df_txt['source_file'].astype(str).map(ttype_map)
        df_txt['Bars'] = df_txt['source_file'].astype(str).map(bars_map)
        df_txt['Tidal'] = df_txt['source_file'].astype(str).map(tidal_map)

        print("TType, Bars and Tidal values successfully assigned!")
        print(f"Number of rows before filtering: {len(df_txt)}")

        print("\n--- Starting filtering of rows with invalid values ---")

        columns_to_filter = ['TType', 'Bars', 'Tidal']

        for col in columns_to_filter:
            df_txt[col] = pd.to_numeric(df_txt[col], errors = 'coerce')
        
        condition_to_remove = df_txt[columns_to_filter].isna().any(axis = 1) | \
                                (df_txt[columns_to_filter] == -999).any(axis = 1)
        
        df_txt_filtered = df_txt[~condition_to_remove].copy()

        print(f"Number of rows after filtering: {len(df_txt_filtered)}")
        print(f"Rows removed: {len(df_txt) - len(df_txt_filtered)}")

        df_txt = df_txt_filtered
        
        print(f"First 5 rows of final DataFrame (after filtering):\n{df_txt.head()}")
        print(f"Columns of final DataFrame: {df_txt.columns.tolist()}")

    except Exception as e:
        print(f"ERROR mapping, assigning or filtering values: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Could not perform processing, as one or both DataFrames were not loaded.")

print("\n" + "="*50 + "\n")

# %%
if df_txt is not None:
    try:
        if 'valor_txt' in df_txt.columns:
            df_txt = df_txt.rename(columns={'valor_txt': 'AGN_ionization'})
        
        df_txt.to_csv(output_txt_file_path, sep='\t', index=False, float_format='%.6f', header=True)
        print(f"File '{output_txt_file_path}' successfully saved using TAB as delimiter!")
    except Exception as e:
        print(f"ERROR saving file with dummies: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Could not save file, as final DataFrame was not created.")

print("\n--- Complete processing finished! ---")


