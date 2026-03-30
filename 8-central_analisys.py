import pandas as pd
import numpy as np

# Caminhos dos arquivos
path_principal = '/scratch/users/giuliano.damian/text/4.5-df_code.txt'
path_spaxel = '/scratch/users/giuliano.damian/text/3-maiores_somas_spaxels.txt'
path_saida = '/scratch/users/giuliano.damian/text/df_code_8.txt'

# 1. Carregar o arquivo principal
print(f"Lendo arquivo principal: {path_principal}")
df = pd.read_csv(path_principal, sep='\t')
# Remove colunas desnecessárias se existirem
df = df.drop(columns=['velocity', 'distance', 'Extinction_r'], errors='ignore')

# 2. Carregar o arquivo Spaxel
print(f"Lendo arquivo spaxel: {path_spaxel}")
df_spaxel = pd.read_csv(path_spaxel, sep='\t')

# --- PADRONIZAÇÃO E LIMPEZA DE NOMES ---

# Renomear para unificar o nome da coluna de busca
if 'CubePrefix' in df_spaxel.columns:
    df_spaxel = df_spaxel.rename(columns={'CubePrefix': 'source_file'})

def limpar_nome(serie):
    """
    Remove prefixos 'manga-', extensões '.fits', '.cube' e espaços vazios
    para garantir que os IDs fiquem idênticos nos dois DataFrames.
    """
    return (serie.astype(str)
            .str.strip()
            .str.replace('manga-', '', regex=False)
            .str.replace('.fits', '', regex=False)
            .str.replace('.cube', '', regex=False))

# Criar colunas temporárias de ID limpo para o Merge
df['id_clean'] = limpar_nome(df['source_file'])
df_spaxel['id_clean'] = limpar_nome(df_spaxel['source_file'])

print(f"Exemplo ID Principal (limpo): {df['id_clean'].iloc[0]}")
print(f"Exemplo ID Spaxel (limpo):    {df_spaxel['id_clean'].iloc[0]}")

# --- MERGE ---
# Selecionamos apenas as colunas de coordenadas do spaxel para o merge
spaxel_cols = [c for c in ['Spaxel_X', 'Spaxel_Y'] if c in df_spaxel.columns]
df_spaxel_min = df_spaxel[['id_clean'] + spaxel_cols]

print("\nRealizando merge baseado nos IDs limpos...")
df = pd.merge(df, df_spaxel_min, on='id_clean', how='left')

# Verificar se houve correspondência
match_count = df['Spaxel_X'].notna().sum()
print(f"Linhas com correspondência encontradas: {match_count}")

if match_count == 0:
    print("!!! ERRO CRÍTICO: Nenhuma correspondência encontrada. Verifique os IDs acima.")
else:
    # 3. FILTRAGEM ESPACIAL
    # Remover linhas que não encontraram par no arquivo spaxel
    df = df.dropna(subset=['Spaxel_X', 'Spaxel_Y']).copy()
    
    # Garantir que as colunas de coordenadas sejam numéricas
    for col in ['Spaxel_X', 'Spaxel_Y', 'x', 'y']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Cálculo da distância quadrática: (Spaxel_X - x)² + (Spaxel_Y - y)²
    df['dist_squared'] = (df['Spaxel_X'] - df['x'])**2 + (df['Spaxel_Y'] - df['y'])**2
    
    # Filtro: Manter apenas o que está dentro do "raio" de 5 pixels (dist² <= 25)
    condicao = df['dist_squared'] <= 25
    df_filtrado = df[condicao].copy()
    
    # Estatísticas
    total_com_match = len(df)
    total_final = len(df_filtrado)
    percentual = (total_final / total_com_match * 100) if total_com_match > 0 else 0
    
    print(f"\n--- Resultado da Seleção Central ---")
    print(f"Total de linhas com dados de spaxel: {total_com_match}")
    print(f"Linhas mantidas no centro (dist² <= 25): {total_final}")
    print(f"Percentual de retenção: {percentual:.2f}%")
    
    # 4. FINALIZAÇÃO E SALVAMENTO
    # Remover colunas auxiliares usadas apenas para o cálculo
    colunas_para_remover = ['id_clean', 'Spaxel_X', 'Spaxel_Y', 'dist_squared']
    df_final = df_filtrado.drop(columns=colunas_para_remover, errors='ignore')
    
    # Salvar o resultado final
    df_final.to_csv(path_saida, sep='\t', index=False)
    print(f"\nArquivo salvo com sucesso em: {path_saida}")
    print(f"Colunas finais ({len(df_final.columns)}): {df_final.columns.tolist()}")

print("\nProcesso concluído!")