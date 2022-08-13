## Primeras acciones de limpieza usualmente presentes:
#   - eliminar registros (filas) duplicados
#   - eliminar registros con nulos en el target
#   - eliminar atributos (columnas) con varianza cercana a cero
#   - eliminar registros con nulos a partir de un nº, estableciendo threshold de valores validos.

import pandas as pd
import numpy as np
from typing import List
from sklearn.feature_selection import VarianceThreshold

import logging

# applic_logger = logging.getLogger('applicLogger')
data_logger = logging.getLogger('dataLogger')



def del_var_zero(df_orig: pd.DataFrame, var_zero_thresh: float = 0,
                 except_cols_varzero: List[str] = None) -> pd.DataFrame:

    """
    Elimina los atributos (columnas) numéricos cuya varianza sea inferior al umbral (threshold) indicado

    param df_orig: dataframe original

    param var_zero_thresh: umbral de varianza

    param except_cols: columnas a excluir (por ejemplo por su definición o naturaleza incumple umbral)

    return: dataframe sin las columnas con varianza inferior al umbral
    """

    df_zero = df_orig.copy()

    # separar numéricas y eliminar las excluidas del análisis
    df_zero_num = df_zero.select_dtypes(include=['number'])
    if except_cols_varzero:
        df_zero_num.drop(columns=except_cols_varzero, inplace=True)

    data_logger.debug(f'VARIANZA CERCANA A CERO, umbral ({var_zero_thresh})\n' +
                      f'columnas a estudiar: {df_zero_num.columns.to_list()}')

    # todas las columnas numéricas a analizar
    cols = set(df_zero_num.columns)

    # se estudia la varianza y se obtienen las que superan umbral mínimo
    vt = VarianceThreshold(threshold=var_zero_thresh)
    _ = vt.fit(df_zero_num)
    mask = vt.get_support()

    # columnas numéricas a eliminar: del total (numéricas sin excluidas) eliminamos las que se deben quedar (mask)
    del_columns = list(cols - set(df_zero_num.columns[mask]))

    if del_columns:
        varz = pd.Series(data=vt.variances_, index=df_zero_num)[~mask]
        data_logger.debug(f'VARIANZA CERCANA A CERO, columnas a eliminar y valores de varianza:\n{varz}')

    data_logger.debug(f'FIN VARIANZA CERCANA A CERO, columnas eliminadas: {del_columns}')

    # del conjunto original (todas, no solo las numéricas) eliminamos las señaladas anteriormente
    if del_columns:
        df_zero.drop(columns=del_columns, inplace=True)

    return df_zero




def first_clean(dataframe: pd.DataFrame, target_name: str, var_zero_thresh: float = 0,
                except_cols_var: List[str] = None, max_null_row: int = 0) -> pd.DataFrame:
    """
    Realiza las primeras tareas de limpieza usuales, tales como eliminar duplicados, registros con nulos en el
    target, columnas con varianza cercana a cero y registros con demasiados nulos (establecido threshold válidas)

    param dataframe: df con los datos de entrada

    param var_zero_thresh: umbral de varianza, atributos numéricos con varianza inferior serán eliminados

    param except_cols_var: columnas a excluir varianza (por ejemplo por su definición o naturaleza incumple umbral)

    param max_null_row: número máximo de nulos permitidos en un registro (fila)

    return: df resultado de las acciones descritas
    """

    df_fclean = dataframe.copy()

    # registros duplicados
    data_logger.debug(f'FIRST CLEAN, registros duplicados: {df_fclean.duplicated().sum()}')
    df_fclean.drop_duplicates(inplace=True)

    # registros con nulos en el target
    data_logger.debug(f'FIRST CLEAN, nulos en el TARGET: {df_fclean[target_name].isnull().sum()}')
    df_fclean.dropna(axis='index', subset=[target_name], inplace=True)

    # atributos con varianza cercana a cero
    df_fclean = del_var_zero(df_orig=df_fclean, var_zero_thresh=var_zero_thresh, except_cols_varzero=except_cols_var)

    # eliminar registros con excesivos nulos, establecido umbral 'max_null_row'
    n_reg = len(df_fclean)
    df_fclean.dropna(thresh=(df_fclean.shape[1] - max_null_row), inplace=True)
    data_logger.debug(f'FIRST CLEAN nº registros con excesivos nulos (max. establecido {max_null_row} nulos):'
                      f'{n_reg - len(df_fclean)} registros')

    return df_fclean
