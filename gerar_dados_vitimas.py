# Gera sinais vitais de vítimas de acidentes
# Este gerador de dados assume independência estatística entre os sinais vitais
# de entrada no momento da amostragem. Em um cenário clínico real, variáveis 
# como lesões vasculares e pressão arterial apresentam interdependência 
# fisiológica.

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

BASE_FOLDER = Path("./datasets/vict/1300v")
BASE_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = BASE_FOLDER / "data.csv"


def calcular_sobr_fisiologica(registro, nivel_ruido=0.02):
    """Calcula a probabilidade de sobrevivência (sobr) baseada as entradas de 1 a 10."""
    # 1. Normalização dos sinais vitais para escala [0, 1] (1.0 = maior gravidade)
    norm_idade = (registro['idade'] - 1) / (90 - 1)
    
    fc = registro['fc']
    norm_fc = 1.0 if fc == 0 else min(abs(fc - 80) / 80.0, 1.0)
    
    fr = registro['fr']
    norm_fr = 1.0 if fr == 0 else min(abs(fr - 16) / 34.0, 1.0)
    
    # PAS Assimétrica (Hipotensão vs Hipertensão)
    pas = registro['pas']
    pas_ideal, pas_hipo, pas_hiper = 120, 80, 200
    if pas == 0:
        norm_pas = 1.0
    elif pas < pas_ideal:
        norm_pas = (pas_ideal - pas) / (pas_ideal - pas_hipo)
    else:
        norm_pas = (pas - pas_ideal) / (pas_hiper - pas_ideal)
    norm_pas = min(max(norm_pas, 0.0), 1.0)
    
    norm_spo2 = (100 - registro['spo2']) / 100.0
    
    # Temperatura Assimétrica (Hipotermia vs Hipertermia)
    temp = registro['temp']
    temp_ideal, temp_hipo, temp_hiper = 36.5, 25.0, 42.0
    if temp < temp_ideal:
        norm_temp = (temp_ideal - temp) / (temp_ideal - temp_hipo)
    else:
        norm_temp = (temp - temp_ideal) / (temp_hiper - temp_ideal)
    norm_temp = min(max(norm_temp, 0.0), 1.0)
    
    norm_pr = 1.0 - registro['pr']           
    norm_sg = registro['sg'] / 3.0           
    norm_fx = float(registro['fx'])          
    norm_queim = registro['queim'] / 3.0     

    # 2. Score Ponderado
    score_gravidade = (
        0.05 * norm_idade +
        0.10 * norm_fc +
        0.10 * norm_fr +
        0.15 * norm_pas +
        0.20 * norm_spo2 +
        0.05 * norm_temp +
        0.15 * norm_pr +
        0.10 * norm_sg +
        0.02 * norm_fx +
        0.08 * norm_queim
    )

    # Adiciona variabilidade individual contínua (suaviza os degraus das variáveis discretas)
    ruido_fisiologico = np.random.normal(0, 0.04)
    score_final = np.clip(score_gravidade + ruido_fisiologico, 0.0, 1.0)
    
    # Probabilidade de sobrevivência
    ruido_fisiologico = np.random.normal(0, 0.04)
    score_final = np.clip(score_gravidade + ruido_fisiologico, 0.0, 1.0)
    
    # Probabilidade de sobrevivência
    sobr = round(float(1.0 - score_final), 2)

    return sobr


def aplicar_ruido_triagem(tri_base, nivel_ruido=0.02):
    """Aplica ruído aleatório trocando a classe da vítima apenas com seus vizinhos diretos."""
    if nivel_ruido > 0 and np.random.random() < nivel_ruido:
        vizinhos = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        return int(np.random.choice(vizinhos[tri_base]))
    return int(tri_base)


def calcular_gcs_e_avpu(sobr, nivel_ruido=0.02):
    """Gera GCS (3–15) e AVPU (0–3) em função de sobr."""
    gcs_base = 3.0 + 12.0 * sobr
    if nivel_ruido > 0:
        ruido_gcs = np.random.normal(0, nivel_ruido * 12.0)
        gcs_val = int(np.clip(round(gcs_base + ruido_gcs), 3, 15))
    else:
        gcs_val = int(round(gcs_base))
        
    if gcs_val >= 15:
        avpu_base = 0
    elif gcs_val >= 12:
        avpu_base = 1
    elif gcs_val >= 8:
        avpu_base = 2
    else:
        avpu_base = 3

    if nivel_ruido > 0 and np.random.random() < nivel_ruido:
        vizinhos = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        avpu_val = int(np.random.choice(vizinhos[avpu_base]))
    else:
        avpu_val = int(avpu_base)

    return gcs_val, avpu_val


def sortear_parametro_num(opcoes_faixas):
    """Sorteia uma faixa e gera um número aleatório dentro dela."""
    faixa = random.choice(opcoes_faixas)
    if faixa[0] == faixa[1]:
        return faixa[0]
    if isinstance(faixa[0], float) or isinstance(faixa[1], float):
        return np.random.uniform(faixa[0], faixa[1])
    return np.random.randint(faixa[0], faixa[1] + 1)


def gerar_dataset_vitimas(distrib_tri={0: 2500, 1: 2500, 2: 2500, 3: 2500}, 
                           media_idade=35, desvio_idade=15, 
                           nivel_ruido=0.02, seed=42):
    """Gera dataset sintético garantindo distribuição uniforme perfeita entre as classes."""
    if seed is not None:
        np.random.seed(seed)   # fixa a semenente no numpy
        random.seed(seed)      # fixa a semente na bib padrão do py

    triagem_parametros = {
        0: {  # Verde (Pico em ~0.93)
            'fc': [(60, 85)], 'fr': [(12, 18)], 'pas': [(110, 130)],
            'spo2': [(96, 100)], 'temp': [(36.1, 37.2)],
            'pr': [1], 'sg': [0], 'fx': [0], 'queim': [0]
        },
        1: {  # Amarelo (Cobre a faixa [0.65, 0.88], Média ~0.78)
            'fc': [(85, 120)], 'fr': [(18, 26)], 'pas': [(90, 109), (131, 155)],
            'spo2': [(88, 95)], 'temp': [(37.2, 38.5), (34.5, 36.0)],
            'pr': [1], 'sg': [0, 1, 2], 'fx': [0, 1], 'queim': [0, 1, 2]
        },
        2: {  # Vermelho (Cobre o centro [0.30, 0.70], Média ~0.51)
            'fc': [(115, 150), (40, 58)], 'fr': [(25, 36), (7, 11)], 
            'pas': [(60, 89), (156, 190)], 'spo2': [(72, 88)], 
            'temp': [(38.2, 40.0), (31.0, 34.4)],
            'pr': [0, 1], 'sg': [1, 2, 3], 'fx': [0, 1], 'queim': [1, 2, 3]
        },
        3: {  # Preto (Cobre a faixa [0.10, 0.38], Média ~0.25)
            'fc': [(0, 39), (151, 180)], 'fr': [(0, 6), (37, 45)], 
            'pas': [(0, 59), (191, 220)], 'spo2': [(50, 71)], 
            'temp': [(25.0, 30.9), (40.1, 42.0)],
            'pr': [0], 'sg': [2, 3], 'fx': [1], 'queim': [2, 3]
        }
    }
    classificacoes_iniciais = []
    for tri_classe, qtd in distrib_tri.items():
        classificacoes_iniciais.extend([tri_classe] * qtd)
    
    random.shuffle(classificacoes_iniciais)
    n_vitimas = len(classificacoes_iniciais)
    idades = np.clip(np.random.normal(media_idade, desvio_idade, n_vitimas), 1, 90).astype(int)

    dados = []

    for i in range(n_vitimas):
        tri_perfil = classificacoes_iniciais[i]
        params = triagem_parametros[tri_perfil]

        def ruido_int(val, min_val, max_val):
            delta = int((max_val - min_val + 1) * nivel_ruido)
            ruido = np.random.randint(-delta, delta+1) if delta > 0 else 0
            return int(np.clip(val + ruido, min_val, max_val))

        def ruido_float(val, min_val, max_val):
            delta = (max_val - min_val) * nivel_ruido
            ruido = np.random.uniform(-delta, delta)
            return float(np.clip(val + ruido, min_val, max_val))

        # Sorteio dos valores base com amostragem bimodal
        fc_base = sortear_parametro_num(params['fc'])
        fr_base = sortear_parametro_num(params['fr'])
        pas_base = sortear_parametro_num(params['pas'])
        spo2_base = sortear_parametro_num(params['spo2'])
        temp_base = sortear_parametro_num(params['temp'])

        # --- 1. Features 1 a 10 ---
        registro = {
            'idade': int(idades[i]),
            'fc': ruido_int(fc_base, 0, 200),
            'fr': ruido_int(fr_base, 0, 50),
            'pas': ruido_int(pas_base, 0, 200),
            'spo2': ruido_int(spo2_base, 0, 100),
            'temp': round(ruido_float(temp_base, 25.0, 43.0), 1),
            'pr': random.choice(params['pr']),
            'sg': random.choice(params['sg']),
            'fx': random.choice(params['fx']),
            'queim': random.choice(params['queim'])
        }

        # --- 2. Target Regressivo (sobr) ---
        registro['sobr'] = calcular_sobr_fisiologica(registro, nivel_ruido=nivel_ruido)

        # --- 3. Features Derivadas (11, 12 e 13) ---
        gcs, avpu = calcular_gcs_e_avpu(registro['sobr'], nivel_ruido=nivel_ruido)
        registro['gcs'] = gcs
        registro['avpu'] = avpu
        
        # Triagem mantendo o equilíbrio de amostragem
        registro['tri'] = aplicar_ruido_triagem(tri_perfil, nivel_ruido=nivel_ruido)

        # Ordenação estrita das colunas
        registro_ordenado = {
            'idade': registro['idade'],
            'fc': registro['fc'],
            'fr': registro['fr'],
            'pas': registro['pas'],
            'spo2': registro['spo2'],
            'temp': registro['temp'],
            'pr': registro['pr'],
            'sg': registro['sg'],
            'fx': registro['fx'],
            'queim': registro['queim'],
            'gcs': registro['gcs'],
            'avpu': registro['avpu'],
            'tri': registro['tri'],
            'sobr': registro['sobr']
        }

        dados.append(registro_ordenado)

    df = pd.DataFrame(dados)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataset salvo com sucesso em: {OUTPUT_CSV}")

    # Exibição das estatísticas
    cores = {0: 'Verde', 1: 'Amarelo', 2: 'Vermelho', 3: 'Preto'}
    contagem = Counter(df['tri'])
    print("\nDistribuição final de vítimas por classificação START (após ruído):")
    for k in sorted(contagem.keys()):
        print(f"  {k} ({cores[k]}): {contagem[k]} ({contagem[k]/n_vitimas*100:.1f}%)")
    
    print(f"total de vítimas: {len(df)}")

    # Histograma de Sobrevivência
    plt.figure(figsize=(8, 4.5))
    plt.hist(df['sobr'], bins=20, color='skyblue', edgecolor='black', weights=np.ones(n_vitimas)/n_vitimas)
    plt.title('Distribuição da Probabilidade de Sobrevivência (sobr)')
    plt.xlabel('Probabilidade de Sobrevivência')
    plt.ylabel('% de Vítimas')
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return df


if __name__ == "__main__":
    distribuicao_desejada = {
        0: 100,  # Verde
        1: 390,  # Amarelo
        2: 405,  # Vermelho
        3: 405   # Preto
    }

    # Com ruído=0, a distribuição será exatamente 25% para cada classe
    # Manter ruído entre 0.02 a 0.05, caso contrário as classes ficam muito
    # misturadas, inclusive as que estão em extremos diferentes
    df_gerado = gerar_dataset_vitimas(
        distrib_tri=distribuicao_desejada,
        media_idade=40,
        desvio_idade=25,
        nivel_ruido=0.05,  
        seed=None
    )