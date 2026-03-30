# %%
import pandas as pd

df = pd.read_csv('/scratch/users/giuliano.damian/text/4.4-df_code.txt', sep = '\t')

new_names = [
    'source_file', 'x', 'y', 'AGN_ionization',
    'xyy', 'xy0', 'xiy',
    'xii', 'xio', 'xo',
    'SFR', 'Av', 'mage_L',
    'Mz_L', 'sigma_star',
    'Sersic_mass','Sersic_n', 'Extinction_r', 'MagAbs_r',
     'TType', 'Bars',
    'Tidal', 'Sersic_b/a',
    'velocity', 'distance'
]

df.columns = new_names

df.to_csv('/scratch/users/giuliano.damian/text/4.5-df_code.txt', sep = '\t', index = False)



