# %%
import pandas as pd
import glob
import os
import numpy as np

main_table_path = "/scratch/users/giuliano.damian/text/4-df_code.txt"
distance_files_txt = "/scratch/users/giuliano.damian/text/3-distancias_txt"
output_file_path = "/scratch/users/giuliano.damian/text/4.1-df_code.txt"

# %%
def load_distance_tables(directory_path):
    distance_dataframes = {}
    pattern = os.path.join(directory_path, "*_distancias_fisicas.txt")
    distance_files = glob.glob(pattern)

    print(f"Distance files found ({len(distance_files)}):")
    if not distance_files:
        print(f"  WARNING: No distance files found with pattern: {pattern}")
        return distance_dataframes
    
    for file_path in distance_files:
        file_name = os.path.basename(file_path)
        prefix = file_name.replace("_distancias_fisicas.txt", "")

        try:
            df = pd.read_csv(file_path, sep = r'\s+', header = 0, dtype = {'x_pixel': int, 'y_pixel': int})
            if df.empty:
                print(f"  WARNING: Distance file {file_name} is empty and will be omitted.")
                continue
            if not {'x_pixel', 'y_pixel', 'distance_kpc'}.issubset(df.columns):
                print(f"  WARNING: File {file_name} does not contain all expected columns (x_pixel, y_pixel, distance_kpc) and will be omitted.")
                continue
            df.set_index(['x_pixel', 'y_pixel'], inplace = True)
            distance_dataframes[prefix] = df
            print(f"  Loaded: {file_name} (for prefix '{prefix}')")
        except pd.errors.EmptyDataError:
            print(f"  WARNING: Distance file {file_name} is empty or malformed and will be omitted.")
        except Exception as e:
            print(f"  ERROR processing distance file {file_name}: {e}")
            
    return distance_dataframes

# %%
def get_distance_kpc_only(row, distance_map, col_source='source_file', col_x='x', col_y='y'):
    id_source = row[col_source]
    
    # Adicionar o prefixo "manga-" ao ID da fonte
    id_source_with_prefix = f"manga-{id_source}"
    
    # Verificar se a fonte com prefixo existe no mapa
    if id_source_with_prefix not in distance_map:
        # Se não encontrar com prefixo, tenta sem prefixo (fallback)
        if id_source not in distance_map:
            return np.nan
        else:
            df_distances = distance_map[id_source]
    else:
        df_distances = distance_map[id_source_with_prefix]
    
    # Converter coordenadas para inteiros
    try:
        coord_x = int(float(row[col_x]))
        coord_y = int(float(row[col_y]))
    except (ValueError, TypeError):
        return np.nan
    
    # Tentar acessar a distância
    try:
        if isinstance(df_distances.index, pd.MultiIndex):
            return df_distances.loc[(coord_x, coord_y), 'distance_kpc']
        else:
            # Se não for MultiIndex, fazer busca booleana
            mask = (df_distances['x_pixel'].astype(float).astype(int) == coord_x) & \
                   (df_distances['y_pixel'].astype(float).astype(int) == coord_y)
            if mask.any():
                return df_distances.loc[mask, 'distance_kpc'].iloc[0]
            return np.nan
    except (KeyError, IndexError):
        return np.nan

# %%
def main():
    if not os.path.exists(main_table_path):
        print(f"ERROR: Main table file not found at '{main_table_path}'!")
        print("Please update the 'main_table_path' variable in the script.")
        return
    print(f"Loading main table from: {main_table_path}")

    try:
        main_df = pd.read_csv(main_table_path, sep = r'\s+')
        print(f"Main table loaded. Number of rows: {len(main_df)}.")

        if main_df.empty:
            print("The main table is empty. There is nothing to process.")
            return
        
        required_columns = ['source_file', 'x', 'y','nsa_sersic_mass']

        if not all(col in main_df.columns for col in required_columns):
            print(f"ERROR: The main table must contain the columns: {', '.join(required_columns)}")
            print(f"Columns found: {main_df.columns.tolist()}")
            return
    except Exception as e:
        print(f"ERROR loading main table: {e}")
        return
        
    print(f"\nLoading distance tables from: {distance_files_txt}")
    
    distance_dfs_map = load_distance_tables(distance_files_txt)

    new_column_name = 'distance_kpc_finded'
    
    if not distance_dfs_map:
        print(f"\nNo distance table was loaded. Column '{new_column_name}' will contain N/A.")
        main_df[new_column_name] = np.nan
    else:
        print(f"\nProcessing main table rows to search and add '{new_column_name}'...")
        print("\nExemplo source_file no main_df:")
        print(main_df['source_file'].unique()[:5])

        print("\nExemplo chaves do distance_dfs_map:")
        print(list(distance_dfs_map.keys())[:5])

        print(main_df[['x','y']].head())
        print(main_df[['x','y']].dtypes)

        # Depois de carregar os DataFrames, adicione este diagnóstico
        print("\n=== DIAGNÓSTICO DETALHADO ===")

        # Verificar os tipos de dados no main_df
        print("TIPOS NO MAIN_DF:")
        print(f"  source_file: {main_df['source_file'].dtype}")
        print(f"  x: {main_df['x'].dtype}")
        print(f"  y: {main_df['y'].dtype}")

        # Mostrar amostra dos dados do main_df
        print("\nAMOSTRA DO MAIN_DF (primeiras 5 linhas):")
        for idx, row in main_df.head(5).iterrows():
            print(f"  Linha {idx}: source='{row['source_file']}', x={row['x']} (tipo: {type(row['x'])}), y={row['y']} (tipo: {type(row['y'])})")

        # Verificar os dados nos arquivos de distância
        print("\nARQUIVOS DE DISTÂNCIA CARREGADOS:")
        for prefix, df in distance_dfs_map.items():
            print(f"\n  Fonte: '{prefix}'")
            print(f"    Shape: {df.shape}")
            print(f"    Tipo do índice: {type(df.index)}")
            
            if isinstance(df.index, pd.MultiIndex):
                print(f"    Níveis do índice: {df.index.names}")
                print(f"    Tipos dos níveis: {df.index.levels[0].dtype}, {df.index.levels[1].dtype}")
                
                # Mostrar amostra do índice
                if len(df) > 0:
                    sample_idx = df.index[0]
                    print(f"    Exemplo de índice: {sample_idx}")
                    print(f"    Tipos no exemplo: {type(sample_idx[0])}, {type(sample_idx[1])}")
            else:
                print(f"    Colunas: {df.columns.tolist()}")
                if len(df) > 0:
                    print(f"    Exemplo de x_pixel: {df['x_pixel'].iloc[0]} (tipo: {type(df['x_pixel'].iloc[0])})")
                    print(f"    Exemplo de y_pixel: {df['y_pixel'].iloc[0]} (tipo: {type(df['y_pixel'].iloc[0])})")
            
            # Mostrar primeiras linhas do arquivo de distância
            if len(df) > 0:
                df_sample = df.head(1)
                print(f"    Primeira linha do arquivo:")
                if isinstance(df.index, pd.MultiIndex):
                    print(f"      Coordenadas: x={df_sample.index[0][0]}, y={df_sample.index[0][1]}")
                else:
                    print(f"      x_pixel: {df_sample['x_pixel'].iloc[0]}, y_pixel: {df_sample['y_pixel'].iloc[0]}")
                print(f"      distance_kpc: {df_sample['distance_kpc'].iloc[0]}")

        # Testar uma linha específica do main_df contra os arquivos de distância
        print("\n=== TESTE DE CORRESPONDÊNCIA ===")
        test_row = main_df.iloc[0]  # Pega a primeira linha para teste
        print(f"Testando com: source='{test_row['source_file']}', x={test_row['x']}, y={test_row['y']}")

        if test_row['source_file'] in distance_dfs_map:
            test_df = distance_dfs_map[test_row['source_file']]
            print(f"Fonte encontrada no mapa!")
            
            # Converter coordenadas de teste para inteiro
            try:
                test_x = int(float(test_row['x']))
                test_y = int(float(test_row['y']))
                print(f"Coordenadas convertidas: ({test_x}, {test_y})")
                
                # Verificar se as coordenadas existem
                if isinstance(test_df.index, pd.MultiIndex):
                    if (test_x, test_y) in test_df.index:
                        print("✓ Coordenadas encontradas no índice!")
                        print(f"  Distância: {test_df.loc[(test_x, test_y), 'distance_kpc']}")
                    else:
                        print("✗ Coordenadas NÃO encontradas no índice")
                        # Mostrar algumas coordenadas do índice para comparação
                        print("  Primeiras 5 coordenadas no índice:")
                        for i, idx in enumerate(test_df.index[:5]):
                            print(f"    {i+1}: ({idx[0]}, {idx[1]})")
                else:
                    # Verificar se as colunas existem
                    mask = (test_df['x_pixel'].astype(float).astype(int) == test_x) & \
                        (test_df['y_pixel'].astype(float).astype(int) == test_y)
                    if mask.any():
                        print("✓ Coordenadas encontradas nas colunas!")
                        print(f"  Distância: {test_df.loc[mask, 'distance_kpc'].iloc[0]}")
                    else:
                        print("✗ Coordenadas NÃO encontradas nas colunas")
            except Exception as e:
                print(f"Erro no teste: {e}")
        else:
            print(f"Fonte '{test_row['source_file']}' NÃO encontrada no mapa")
            print(f"Fontes disponíveis: {list(distance_dfs_map.keys())}")


        main_df[new_column_name] = main_df.apply(
            get_distance_kpc_only,
            axis = 1,
            distance_map = distance_dfs_map,
            col_source = 'source_file',
            col_x = 'x',
            col_y = 'y' 
        )
    
    values_found = main_df[new_column_name].notna().sum()
    print(f"\nValues for '{new_column_name}' were found in {values_found} out of {len(main_df)} rows.")
    if values_found == 0 and len(main_df) > 0:
        print("  ALERT: No 'distance_kpc' value was found for any row.")

    sersic_mass_col_name = 'nsa_sersic_mass'

    if sersic_mass_col_name in main_df.columns:
        print(f"\nTransforming column '{sersic_mass_col_name}' to log10...")

        nan_before_numeric = main_df[sersic_mass_col_name].isnull().sum()

        main_df[sersic_mass_col_name] = pd.to_numeric(main_df[sersic_mass_col_name], errors = 'coerce')

        nan_after_numeric = main_df[sersic_mass_col_name].isnull().sum()

        if nan_after_numeric > nan_before_numeric:
            print(f"  WARNING: {nan_after_numeric - nan_before_numeric} non-numeric value(s) in '{sersic_mass_col_name}' were converted to NaN.")

        log_values = np.log10(main_df[sersic_mass_col_name])

        inf_count = np.isinf(log_values).sum()

        if inf_count > 0:
            print(f"  WARNING: {inf_count} value(s) in '{sersic_mass_col_name}' resulted in infinity (probably from original values equal to zero).")
        
        nan_after_log = log_values.isnull().sum()
        if nan_after_log > nan_after_numeric:
            print(f"  WARNING: {nan_after_log - nan_after_numeric} negative value(s) in '{sersic_mass_col_name}' resulted in NaN after applying log10.")

        main_df[sersic_mass_col_name] = log_values
        print(f"Column '{sersic_mass_col_name}' replaced with its log10 values.")
    else:
        print(f"\nWARNING: Column '{sersic_mass_col_name}' was not found in the main table. The log10 transformation will not be applied.")
    
    print(f"\nSaving combined table to: {output_file_path}")

    try:
        main_df.to_csv(output_file_path, sep = '\t', index = False, na_rep = 'N/A')
        print("Process completed!")
        print(f"The final table with modifications was saved at '{output_file_path}'.")
    except Exception as e:
        print(f"ERROR saving combined table: {e}")

# %%
if __name__ == '__main__':
    main()


