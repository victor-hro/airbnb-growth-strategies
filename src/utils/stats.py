import numpy as np
from scipy import stats

def calculate_quartiles_and_iqr(data):
    """
    Calcula o primeiro quartil (Q1), o terceiro quartil (Q3) e a amplitude interquartil (IQR) de um conjunto de dados.

    Args:
        data (numpy array or list): O conjunto de dados.

    Returns:
        Q1 (float): O primeiro quartil.
        Q3 (float): O terceiro quartil.
        IQR (float): A amplitude interquartil.
    """
    # Ordena o conjunto de dados
    sorted_data = np.sort(data)

    # Calcula o primeiro quartil (Q1)
    Q1 = np.percentile(sorted_data, 25)

    # Calcula o terceiro quartil (Q3)
    Q3 = np.percentile(sorted_data, 75)

    # Calcula a amplitude interquartil (IQR)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Imprime os valores calculados
    print(f"Primeiro Quartil (Q1): {Q1}")
    print(f"Terceiro Quartil (Q3): {Q3}")
    print(f"Amplitude Interquartil (IQR): {IQR}")
    print(f"Limite inferior: {limite_inferior}")
    print(f"Limite superior: {limite_superior}")

    return Q1, Q3, IQR, limite_inferior, limite_superior



def remove_outliers_iqr(df, column_name):
    df_copy = df.copy()
    """
    Remove outliers de um DataFrame com base na regra do IQR (Intervalo Interquartil).

    Parâmetros:
    - df: O DataFrame a ser processado.
    - column_name: O nome da coluna na qual os outliers serão identificados e removidos.

    Retorna:
    - Um novo DataFrame com os outliers removidos.
    """
    Q1, Q3, IQR, limite_inferior, limite_superior = calculate_quartiles_and_iqr(df_copy[column_name])

    # Remova as linhas que contêm outliers
    df_result = df_copy[(df_copy[column_name] >= limite_inferior) & (df_copy[column_name] <= limite_superior)]

    return df_result


def t_test(group1, group2, alpha=0.05):
    """
    Realiza um teste t independente para comparar as médias de dois grupos.

    Args:
        group1 (pd.Series): O primeiro grupo de dados.
        group2 (pd.Series): O segundo grupo de dados.
        alpha (float): O nível de significância. Padrão é 0.05.

    Returns:
        dict: Um dicionário contendo o resultado do teste.
    """
    # Realize o teste t independente
    t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    # Determine se a diferença nas médias é estatisticamente significativa
    if p_value < alpha:
        result = {
            'test_result': 'Significativo',
            'test_statistic': t_statistic,
            'p_value': p_value,
            'alpha': alpha
        }
    else:
        result = {
            'test_result': 'Não Significativo',
            'test_statistic': t_statistic,
            'p_value': p_value,
            'alpha': alpha
        }

    return result
