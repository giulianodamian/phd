# %%
import pandas as pd
import numpy as np

ORIGINAL_DATA_PATH = '/scratch/users/giuliano.damian/text/4.2-df_code.txt'
DEPROJECTION_PATH = '/scratch/users/giuliano.damian/text/4.3.1-df_code.txt'
FINAL_DATA_PATH = '/scratch/users/giuliano.damian/text/4.4-df_code.txt'


# %%
def create_final_data(original_data_path, deprojection_path, final_data_path):
    try:
        print('Reading data files...')
        df_original = pd.read_csv(original_data_path, sep = '\t')
        df_deprojection = pd.read_csv(deprojection_path, sep = '\t')

        df_original.dropna(subset = ['source_file'], inplace = True)

        print('\nRemoving columns that exist in both files to avoid duplicates...')

        columns_to_remove_original = ['nsa_sersic_ba']
        df_original.drop(columns = columns_to_remove_original, inplace = True, errors = 'ignore')

        merge_keys = ['source_file', 'x', 'y']
        print('\nMerging with ', merge_keys)

        df_final = pd.merge(df_original, df_deprojection, on = merge_keys, how = 'inner')

        print(f'\nMerge Done! {len(df_final)} lines found!')

        print('Cleanig data and removing unnecessary columns...')
        df_final.replace(-999, np.nan, inplace = True)

        columns_to_remove_final = ['dist_real', 'dist_pixel', 'y_rot', 'x_rot', 'center_x', 'center_y', 'angle_phi', 'angle_alpha', 'angle_theta', 'angle_i', 'vrot_star', 'distance_kpc_finded']
        
        df_final.drop(columns = columns_to_remove_final, inplace = True, errors = 'ignore')
        print('\nColumns removed:', columns_to_remove_final)

        df_final.dropna(inplace = True)
        print(f'After cleaning we have {len(df_final)} lines.')

        cols = df_final.columns.tolist()

        if 'source_file' in cols:
            cols.insert(0, cols.pop(cols.index('source_file')))
            df_final = df_final[cols]

        print('\nFinal Columns to be saved:', df_final.columns.tolist())

        df_final.to_csv(final_data_path, sep = '\t', index = False)
        return True
    except Exception as e:
        print(f'Critical ERROR on processing: {e}')
        return False

# %%
if __name__ == '__main__':
    create_final_data(ORIGINAL_DATA_PATH, DEPROJECTION_PATH, FINAL_DATA_PATH)


