## MODULO PARA CARGAR LOS DATOS Y DAR FORMATO A LAS COLUMNAS: corregir nombre, reordenar, renombrar, asignar tipo,
## asignar indices

# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Varios
# ==============================================================================
import os
from typing import Dict, List


from src.custom_preprox.first_clean import first_clean
from src.custom_preprox.verify_columns_values import verify_num_column_values

import logging

applic_logger = logging.getLogger('applicLogger')
data_logger = logging.getLogger('dataLogger')


# CONFIGURACION
# ==============================================================================

col_list = ['Country', 'Year', 'Status', 'Population', 'Income_composition_of_resources', 'Total_expenditure',
            'percentage_expenditure', 'GDP', 'Hepatitis_B', 'Measles', 'Polio', 'Diphtheria', 'HIV_AIDS', 'BMI',
            'Alcohol', 'thinness_5_9_years', 'thinness__1_19_years', 'Schooling', 'under_five_deaths',
            'infant_deaths', 'Adult_Mortality', 'Life_expectancy']
cols_rename = {'Total_expenditure': 'pct_total_exp', 'percentage_expenditure': 'pct_exp_GDP',
               'Income_composition_of_resources': 'Income_index'}
cols_type = {'category': ['Country', 'Status']}
col_idx = ['Country', 'Year']

target_name = 'Life_expectancy'
var_zero_thresh = 0.5
except_cols_var = ['Income_index']
max_null_row = 5

del_columns = ['Population', 'Schooling', 'under_five_deaths', 'thinness_5_9_years']

col_def_rules = {'pct': ['pct_exp_GDP', 'pct_total_exp', 'Hepatitis_B', 'Polio', 'Diphtheria', 'thinness_5_9_years',
                         'thinness__1_19_years'],
                 'x1000': ['Adult_Mortality', 'infant_deaths', 'Measles', 'under_five_deaths', 'HIV_AIDS'],
                 '0_1': ['Income_index']}
del_wrong_thresh_col = True
max_wrong = 0.20
assign_na = True


# ==============================================================================


def correct_columns_name(names_list: List[str]) -> List[str]:
    """
    Elimina espacios y sustituye otros caracteres separadores por carácter 'subrayado'.

    param names_list: lista con los nombres de las columnas.

    return: lista de nombres corregida
    """

    names_list = [col.strip() for col in names_list]
    names_list = [col.replace(' ', '_') for col in names_list]
    names_list = [col.replace('-', '_') for col in names_list]
    names_list = [col.replace('/', '_') for col in names_list]

    return names_list


def format_columns(dataframe: pd.DataFrame, order_cols: List[str], rename_cols: dict[str, str],
                   dict_col_type: Dict[str, List[str]], cols_index: List[str]) -> pd.DataFrame:
    """
    Función que da formato a los nombres de las columnas del dataset (quita espacios, caracteres no deseados..),
    reordena las mismas, cambia algunos nombres por otros más descriptivos, reasigna en su caso tipos adecuados a las
    columnas que lo necesiten y crea un índice con los atributos especificados si asi se dispone.

    param dataframe: dataframe con los datos originales

    param order_cols: lista con el orden de las columnas deseado (EDA, semántica o relación atributos..). Puede servir
    también para eliminar de partida algunos atributos (basta con no contemplarlas en la lista)

    param rename_cols: dict con claves nombre antiguo y valores nombre nuevo.

    param dict_col_type: dict con claves los tipos y valores listas con las columnas que se corresponden con
    el tipo especificado. Muy usual asignar tipo correcto a categóricas y corregir otras asignaciones erróneas.

    param cols_index: Lista especificando columnas que forman el índice (EDA, series temporales, etc)

    NOTA: Este orden al describir los parámetros es el que se sigue en las operaciones

    return: dataframe con las columnas resultado del formato
    """

    dataf = dataframe.copy()

    # corrección (formato) nombres de las columnas
    dataf.columns = correct_columns_name(dataf.columns.to_list())

    # reordenar columnas
    if order_cols:
        dataf = dataf[order_cols]

    # renonbrar columnas en su caso
    if rename_cols:
        dataf.rename(columns=rename_cols, inplace=True)

    # asignar tipo correcto
    for coltype, cols in dict_col_type.items():
        for col in cols:
            dataf[col] = dataf[col].astype(coltype)

    # asignamos índice en su caso
    if cols_index:
        dataf.set_index(cols_index, inplace=True)

    return dataf


def etl_data(origin: str) -> pd.DataFrame:
    """
    Carga los datos originales, les aplica formato a las columnas (nombres, orden, indice) y reliza una primera
    limpieza basica

    param origin: ruta datos originales

    return: dataframe con el formato aplicado
    """

    applic_logger.info('INICIO ETL')
    data_logger.info('============== INICIO ETL_DATA ==============')

    # carga dataset del archivo csv
    data_logger.info('Recuperando datos de ORIGEN')
    dataf = pd.read_csv(origin)
    data_logger.info('Carga datos ORIGEN completada')

    data_logger.info('INICIO formato de columnas y limpieza básica')

    # formato columnas
    dataf = format_columns(dataframe=dataf, order_cols=col_list, rename_cols=cols_rename, dict_col_type=cols_type,
                           cols_index=col_idx)

    # limpieza basica dataset
    dataf = first_clean(dataframe=dataf, target_name=target_name, var_zero_thresh=var_zero_thresh,
                        except_cols_var=except_cols_var, max_null_row=max_null_row)

    data_logger.info('FIN formato de columnas y limpieza básica')

    # eliminar columnas (EDA: correladas, nº elevado valores erróneos o nulos, etc.)
    dataf.drop(columns=del_columns, inplace=True)

    data_logger.info('EDA: columnas eliminadas (correladas, nº elevado valores erróneos o nulos, etc.):')
    data_logger.info(del_columns)


    # verificar y corregir columnas numéricas según sus reglas establecidas
    if col_def_rules:

        data_logger.info('INICIO verificación de valores columnas numéricas')

        dataf = verify_num_column_values(dataframe=dataf, dict_col_rules=col_def_rules,
                                         del_wrong_thresh_col=del_wrong_thresh_col, max_wrong_col=max_wrong,
                                         assign_na=assign_na, show_results=False)

        data_logger.info('FIN verificación de valores columnas numéricas')


    # AÑADIDOS FUERA CONFIGURACION (CASOS PARTICULARES, ETC)
    # EDA: valores de 'Income_index' igual a cero por no disponibilidad, asignamos valor nulo
    dataf.loc[dataf['Income_index'] == 0, 'Income_index'] = np.nan

    data_logger.debug('EDA: se decide tomar como nulos los valores cero para el atributo "Income_index"')

    applic_logger.info('FIN ETL')
    data_logger.info('============== FIN ETL_DATA ==============')

    return dataf


if __name__ == "__main__":
    # ejplo carga archivo entrenamiento (deberá indicarse en CONFIG en el módulo que se use)
    DATA_FOLDER = '../data'
    FILENAME = 'Life Expectancy Data.csv'
    DATAFILE = os.path.join(DATA_FOLDER, FILENAME)

    df = etl_data(DATAFILE)
    print('CARGA DATOS')

    df.info()
