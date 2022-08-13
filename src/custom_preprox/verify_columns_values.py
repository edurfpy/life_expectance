#
# Comprueba que los datos de las columnas se encuentren dentro del rango válido por definición (unidades, no confundir
# con outliers). Hablamos de porcentajes o cantidades por mil o diez mil habitantes, como algunos ejemplos.

# Localizados esos errores por atributo (columna), se presentan las opciones de mostrar tabla con la información, además
# de dado un umbral máximo de errores por columna eliminar aquellas que lo superen o si están por debajo de ese umbral
# asignar a dichos erróneos NA para posterior tratamiento.
#
# La regla para cada columna se 'codifica' como una cadena, de entre las siguientes:
#       - 'pct': porcentaje
#       - 'pos', 'neg': positivo y negativo, respectivamente (incluyen el valor cero)
#       - 'x<valor>', por ejemplo 'x1000': para magnitudes demográficas o que se definan sobre un total determinado,
#               por ejemplo afectados por cada mil habitantes.
#       - caso genérico '<min>_<max>', por ejemplo '0_17': valores en ese intervalo, incluidos los extremos descritos
#
#

import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple
from src.custom_preprox.custom_transf_preprox import AssignWrongValuesTransformer

import logging

# applic_logger = logging.getLogger('applicLogger')
data_logger = logging.getLogger('dataLogger')


def rule_lims(rule: str) -> Union[None, Tuple[float, float]]:
    if rule == 'pct':
        r_min, r_max = 0, 100

    elif rule == 'pos':
        r_min, r_max = 0, np.PINF

    elif rule == 'neg':
        r_min, r_max = np.NINF, 0

    elif rule.startswith('x'):
        r_min = 0
        r_max = float(rule[1:])

    # CASO: regla valores mínimo y máximo, separados por carácter guión bajo '_'
    elif rule.find('_') != -1:
        lims = rule.split('_')

        if len(lims) != 2:
            print(f'Error en la especificación de la regla: {rule}',
                  'La regla debe constar de los valores mínimo y máximo del rango válido, separados por un '
                  'guión bajo <_>', sep='\n')
            return None

        r_min, r_max = float(lims[0]), float(lims[1])

    # CASO: la regla no se corresponde con ninguna de las definidas
    else:
        print(f'Error en la especificación de la regla: {rule}',
              'Regla mal descrita, no se corresponde con las disponibles', sep='\n')
        return None

    return r_min, r_max


def verify_num_column_values(dataframe: pd.DataFrame, dict_col_rules: Dict[str, List[str]],
                             show_results: bool = True, del_wrong_thresh_col: bool = False,
                             max_wrong_col: Union[int, float] = 0.25,
                             assign_na: bool = False) -> Union[None, pd.DataFrame]:

    # inicializaciones necesarias
    df_verfy = dataframe.copy()
    dict_total_wrong: Dict[str, int] = dict()
    dict_idx_col_assign: Dict[str, List[int]] = dict()

    if 0 <= max_wrong_col < 1:
        max_wrong_col = np.ceil(len(df_verfy) * max_wrong_col)

    data_logger.debug(f'VERIFY NUM COLS, Umbral {max_wrong_col} erróneos de total {len(df_verfy)}')

    # búsqueda de valores erróneos en las columnas
    for rule, cols in dict_col_rules.items():

        data_logger.debug(f'VERIFY NUM COLS, regla {rule} aplicada a columnas {cols}')

        # asignamos los límites correspondientes de la regla en cuestión, func. rule_lims()
        if rule_lims(rule):
            col_min, col_max = rule_lims(rule)
        else:
            # caso error, regla no valida, etc
            continue

        # contabilizamos para las columnas especificadas los registros erróneos (índice) y su nº total
        for col in cols:

            # columnas eliminadas en algún paso anterior
            # (de esta forma se permite definir el diccionario de reglas desde el principio)
            if col not in df_verfy.columns.tolist():
                continue

            idx_wrong = df_verfy.loc[(df_verfy[col] < col_min) | (df_verfy[col] > col_max), col].index.tolist()

            if idx_wrong:
                data_logger.debug(f'VERIFY NUM COLS, Columna {col}: {len(idx_wrong)} registro(s) erróneo(s)')

            dict_total_wrong.update({col: len(idx_wrong)})

            if assign_na & (0 < len(idx_wrong) <= max_wrong_col):
                dict_idx_col_assign.update({col: idx_wrong})

    # pasamos totales a pandas series (presentación, mejor manejo)
    wrong_cols_serie = pd.Series(dict_total_wrong)

    # CASO: opción mostrar totales valores erróneos (info, EDA...)
    if show_results:
        print(f'Valores erróneos (no coherentes con unidades o definición) por columna, umbral erróneos {max_wrong_col}')
        print(pd.DataFrame({'#n': wrong_cols_serie, '%': wrong_cols_serie/len(df_verfy)*100}))
        if (not del_wrong_thresh_col) and (not assign_na):
            return None

    # CASO: opción eliminar columnas con total de erróneos por encima del umbral especificado
    if del_wrong_thresh_col:
        del_cols = list(wrong_cols_serie[wrong_cols_serie > max_wrong_col].index)
        df_verfy.drop(columns=del_cols, inplace=True)

        data_logger.debug(f'VERIFY NUM COLS, Eliminadas por exceso erróneos las columnas: {del_cols}')


    # CASO: cuando el total de erróneos en columnas es inferior o igual al máximo, opción de asignar nulos para
    # posterior tratamiento

    if assign_na:
        assign_transf = AssignWrongValuesTransformer(dict_cols=dict_idx_col_assign, value=np.nan)
        df_verfy = assign_transf.fit_transform(df_verfy)

        data_logger.debug(f'VERIFY NUM COLS, Columnas con erróneos inferior umbral, se asignan como nulos:')
        data_logger.debug(f'{list(dict_idx_col_assign.keys())}')

    # devolvemos dataframe corregido
    return df_verfy


if __name__ == '__main__':
    datos = pd.DataFrame({'A': [25, 42, 70, 120, -3], 'B': [120, 256, 842, 1008, 900], 'C': [-14, -98, -75, 20, 40],
                          'D': [12, 15, 7, 6, 20]})

    dict_rules = {'pct': ['A'], 'x1000': ['B'], 'neg': ['C'], '0_18': ['D']}

    df = verify_num_column_values(dataframe=datos, dict_col_rules=dict_rules, del_wrong_thresh_col=True,
                                  max_wrong_col=1, assign_na=True)

    print(df)
