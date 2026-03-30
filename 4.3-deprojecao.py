# %%
import os
import astropy.io.fits as fits
import numpy as np
import pandas as pd

TXT_DATA_DIR = '/scratch/users/giuliano.damian/text'
CUBES_FITS_DIR = '/data/public/sdss/manga/mcubes'
INPUT_TXT_1 = os.path.join(TXT_DATA_DIR, '4.2-df_code.txt')

INPUT_TXT_1_COLS = ['source_file', 'x', 'y', 'nsa_sersic_ba', 'distance_kpc_finded']

OUTPUT_TXT = os.path.join(TXT_DATA_DIR, '4.3-df_code.txt')
COORDINATES_FILES = os.path.join(TXT_DATA_DIR, '3-maiores_somas_spaxels.txt')
DRPALL_PATH = '/scratch/users/giuliano.damian/data/other_data/drpall-v3_1_1.fits'
DELIMITER = '\t'

df = pd.read_csv(INPUT_TXT_1, sep = DELIMITER, usecols = INPUT_TXT_1_COLS)

df_coords = pd.read_csv(COORDINATES_FILES, sep = DELIMITER)

df_coords.rename(columns={
            'CubePrefix': 'source_file', 
            'Spaxel_X': 'center_x', 
            'Spaxel_Y': 'center_y'
        }, inplace=True)
df_coords['source_file'] = df_coords['source_file'].str.replace('manga-', '', regex=False)

df = pd.merge(df, df_coords[['source_file', 'center_x', 'center_y']], on = 'source_file', how = 'left')

properties = {}
with fits.open(DRPALL_PATH) as hdul:
    data = hdul[1].data
    for i in range(len(data)):
        plateifu = data['PLATEIFU'][i].strip()
        properties[plateifu] = {'angle_phi': data['nsa_sersic_phi'][i]}

df_drpall = pd.DataFrame.from_dict(properties, orient = 'index').reset_index().rename(columns = {'index': 'source_file'})
df_drpall['angle_phi'] = df_drpall['angle_phi'] - 90.


df = pd.merge(df, df_drpall, on = 'source_file', how = 'left')

cols_to_numeric = ['x', 'y', 'center_x', 'center_y', 'nsa_sersic_ba', 'angle_phi']
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors = 'coerce')

delta_x = df['x'] - df['center_x']
delta_y = df['y'] - df['center_y']

df['angle_alpha'] = np.arctan2(delta_y, delta_x)
df['angle_alpha'] = np.rad2deg(df['angle_alpha'])

df['angle_theta'] = df['angle_alpha'] - df['angle_phi']

b_over_a = df['nsa_sersic_ba']
condition = (b_over_a >=0) & (b_over_a <=1)

df['angle_i'] = np.arcsin(b_over_a)
df['angle_i'] = np.rad2deg(df['angle_i'])
df.loc[~condition, 'angle_i'] = np.nan

vrot_maps = {}
filenames = [f for f in os.listdir(CUBES_FITS_DIR) if f.endswith('-MEGACUBE.fits')]

for filename in filenames:
    plateifu = '-'.join(filename.split('-')[1:-1])
    try:
        file_path = os.path.join(CUBES_FITS_DIR, filename)
        with fits.open(file_path) as hdul:
            if 'PoPBins' in hdul and hdul['PoPBins'].data.ndim == 3 and hdul['PoPBins'].data.shape[0] > 31:
                vrot_maps[plateifu] = hdul['PoPBins'].data[31]
    except Exception as e:
        print(f'Error processing {filename}: {e}')

print(f'[OK] Loaded {len(vrot_maps)} VROT maps.')


# %%
def get_vrot_value(row, maps):
    vrot_map = maps.get(row['source_file'])
    if vrot_map is not None:
        try:
            x_coord = int(round(row['x']))
            y_coord = int(round(row['y']))

            if 0 <= y_coord < vrot_map.shape[0] and 0 <= x_coord < vrot_map.shape[1]:
                return vrot_map[y_coord, x_coord]
        except (ValueError, IndexError):
            pass
    return np.nan

# %%
if vrot_maps:
    df['vrot_star'] = df.apply(get_vrot_value, maps = vrot_maps, axis = 1)
else:
    df['vrot_star'] = np.nan
    print("WARNING: No VROT maps were loaded.")


# %%
num_original_rows = len(df)
critical_columns = ['angle_phi', 'angle_alpha', 'angle_theta', 'angle_i', 'vrot_star']

df.dropna(subset = critical_columns, inplace = True)

num_final_rows = len(df)
num_removed_rows = num_original_rows - num_final_rows
print(f"Analysis complete: {num_removed_rows} rows removed.")


# %%
columns_to_add = ['y_rot', 'x_rot', 'center_y_rot', 'center_x_rot', 'v_real', 'real_dist']
for col in columns_to_add:
    df[col] = np.nan

# %%
df.to_csv(OUTPUT_TXT, sep = DELIMITER, index = False, na_rep = 'NaN')

print(f"\nFINAL OPERATION COMPLETED! File '{OUTPUT_TXT}' was generated successfully.")


