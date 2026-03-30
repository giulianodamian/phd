import pandas as pd

df = pd.read_csv('/scratch/users/giuliano.damian/text/4.5-df_code.txt', sep='\s+', engine='python')

df.columns = df.columns.str.strip()

filtro_agn = df[df['AGN_ionization'] == 1]
total_galaxias_agn = filtro_agn['source_file'].nunique()

print(f"Total de galáxias com ao menos 1 spaxel AGN: {total_galaxias_agn}")
