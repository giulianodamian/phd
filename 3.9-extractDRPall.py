# %%
from astropy.table import Table
import pandas as pd

# ==============================
# CAMINHOS
# ==============================

input_fits = "/scratch/users/giuliano.damian/data/other_data/drpall-v3_1_1.fits"
output_txt = "/scratch/users/giuliano.damian/data/other_data/manga_data_drpall_extracted.txt"

print("Iniciando extração completa + nsa_extinction_r...")

# ==============================
# 1. LER FITS (tabela principal)
# ==============================

dat = Table.read(input_fits, format='fits', hdu=1)

# ==============================
# 2. MANTER TODAS AS COLUNAS 1D (como você já fazia)
# ==============================

valid_columns = [name for name in dat.colnames if len(dat[name].shape) <= 1]
df = dat[valid_columns].to_pandas()

# ==============================
# 3. ADICIONAR APENAS A BANDA r
# Ordem padrão:
# 0=FUV, 1=NUV, 2=u, 3=g, 4=r, 5=i, 6=z
# ==============================

df["nsa_extinction_r"] = dat["nsa_extinction"][:, 4]
df['nsa_sersic_absmag_r'] = dat['nsa_sersic_absmag'][:, 4]
# ==============================
# 4. DECODIFICAR BYTES
# ==============================

for col in df.select_dtypes([object]):
    df[col] = df[col].apply(
        lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else x
    )

# ==============================
# 5. SALVAR NO MESMO FORMATO
# ==============================

with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("# " + " ".join(df.columns) + "\n")
    df.to_csv(f, sep=' ', index=False, header=False)

print("Arquivo atualizado com sucesso!")
print(f"Coluna adicionada: nsa_extinction_r")
print(f"Total de colunas agora: {len(df.columns)}")


