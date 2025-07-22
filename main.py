import pandas as pd
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
from tqdm import tqdm


# Definição de função para abertura de todas worksheets dentro do arquivo xlsx
def read_all_sheets_into_one_df(file_path):
    # Leitura de todas planilhas dentro do excel
    all_sheets = pd.read_excel(file_path, sheet_name=None)

    # Concatenar todos dataframes do dicionario em um
    combined_df = pd.concat(all_sheets.values(), ignore_index=True)

    return combined_df

#%%
_bytree = read_all_sheets_into_one_df("IFC_2015_2024_ByTree_v01.xlsx")

#%%
df = _bytree.copy()

#%%
# --- Definir intervalo de classe (IC) e criar coluna de classe central ---
IC = 2  # intervalo de classe em cm
df['Classe_DAP'] = (np.floor(df['COVADAP'] / IC) * IC) + (IC / 2)

# --- Limpar COVADAP: remover NaN e zeros ---
df_cleaned = df[(df['COVADAP'].notna()) & (df['COVADAP'] > 0)]

# Criar colunas para parâmetros
df['weibull_shape'] = np.nan
df['weibull_loc'] = np.nan
df['weibull_scale'] = np.nan

# --- Loop com barra de progresso ---
for id_val in tqdm(df_cleaned['ID'].unique(), desc="Ajustando Weibull por ID"):
    df_id = df_cleaned[df_cleaned['ID'] == id_val]

    # Frequência por Classe_DAP
    frequencia = df_id['Classe_DAP'].value_counts().sort_index()
    x = frequencia.index.values
    y = frequencia.values

    if len(x) < 2:
        continue

    dados_expandido = np.repeat(x, y)

    try:
        shape, loc, scale = weibull_min.fit(dados_expandido, floc=0)
    except Exception as e:
        print(f"Erro no ajuste para ID {id_val}: {e}")
        continue

    df.loc[df['ID'] == id_val, 'weibull_shape'] = shape
    df.loc[df['ID'] == id_val, 'weibull_loc'] = loc
    df.loc[df['ID'] == id_val, 'weibull_scale'] = scale

#%%
max_rows = 1048576  # Maximum number of rows Excel supports

# Create a Pandas Excel writer using XlsxWriter as the engine.
with pd.ExcelWriter("ajustado.xlsx", engine='xlsxwriter') as writer:
    # Split DataFrame into chunks and save each chunk to a separate sheet
    for i in range(0, len(df), max_rows):
        chunk = df.iloc[i:i + max_rows]
        chunk.to_excel(writer, sheet_name=f'WBL_{i // max_rows + 1}', index=False)

print("Data has been successfully saved to multiple sheets.")