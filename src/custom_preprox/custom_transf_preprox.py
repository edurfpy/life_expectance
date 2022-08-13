#
#
#
#
from typing import Dict, List

import numpy as np
import pandas as pd
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder


# =======================================================================================================

# Transformer customizado (scikit) para asignar nuevo valor en columnas con valores no coherentes con su definición
# o con las unidades en las que se define (por ejemplo valores de porcentaje mayores de 100, valores superiores a mil
# en magnitudes por mil habitantes...).
#
# Diferenciar del caso de imputar nulos, para el cual ya hay tratamiento disponible (métodos 'fullna', transformadores
# familia 'Imputer')
#
# A la clase se le proporciona un diccionario con claves el nombre de la columna y valor lista con los índices (filas)
# cuyos valores se remplazarán en dicha columna.
#
# Se ha separado del módulo 'verify_columns_values', caso sustituir, por su posible utilidad como transformador en
# otros posibles escenarios.


class AssignWrongValuesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, dict_cols: Dict[str, List[int]], value: float):
        self.dict_col = dict_cols
        self.value = value

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X_copy = X.copy()

        for col, idx in self.dict_col.items():
            X_copy.loc[idx, col] = self.value

        return X_copy


# =======================================================================================================

#
#
#
#

class CustomLexpectPreprx(TransformerMixin, BaseEstimator):

    def __init__(self, pwr_transf: bool = False, ret_df: bool = False):
        self.power_transf_ = pwr_transf
        self.transformer_ = None
        self.return_df_ = ret_df
        self.__categoric_cols = None
        self.__numeric_cols = None

    def __create_preprx_transformer(self):
        num_steps = [('imputer', IterativeImputer(random_state=22))]
        if self.power_transf_:
            num_steps.append(('pow_transf', PowerTransformer(method='yeo-johnson', standardize=True)))

        numeric_transformer = Pipeline(steps=num_steps)

        categ_transformer = OneHotEncoder(drop='if_binary')

        self.transformer_ = ColumnTransformer(transformers=[('numeric', numeric_transformer,
                                                             make_column_selector(dtype_include='number')),
                                                            ('category', categ_transformer,
                                                             make_column_selector(dtype_exclude='number'))],
                                              remainder='passthrough')

    def fit(self, X, y=None):
        Xcopy = X.copy()

        if isinstance(Xcopy, pd.DataFrame):
            self.__numeric_cols = Xcopy.select_dtypes(include='number').columns.to_list()
            self.__categoric_cols = Xcopy.select_dtypes(include='category').columns.to_list()

        self.__create_preprx_transformer()
        self.transformer_.fit(Xcopy)

        return self

    def transform(self, X):
        Xcopy = X.copy()
        Xtransf = self.transformer_.transform(Xcopy)

        if self.return_df_:

            if isinstance(Xcopy, pd.DataFrame):
                encode_categ_labels = self.transformer_.named_transformers_. \
                    category.get_feature_names_out(self.__categoric_cols)
                labels_col = np.concatenate([self.__numeric_cols, encode_categ_labels])

                Xtransf = pd.DataFrame(data=Xtransf, columns=labels_col)

            else:
                print('WARNING:', 'No se puede devolver resultado como dataframe. Necesita que la entrada sea de este \
                mismo tipo para obtener información de las columnas', sep='\n', end='\n' * 2)

        return Xtransf
