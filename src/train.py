#
#
#
#

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd

# # Gráficos
# # ==============================================================================
# import matplotlib.pyplot as plt
# import sklearn.base
# from matplotlib import style
# import matplotlib.ticker as ticker
# import seaborn as sns

# # Configuración matplotlib
# # ==============================================================================
# plt.rcParams['image.cmap'] = "bwr"
# # plt.rcParams['figure.dpi'] = "100"
# plt.rcParams['savefig.bbox'] = "tight"
# style.use('ggplot') or plt.style.use('ggplot')

# Modelado (scikit)
# ==============================================================================
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
# from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

import joblib

# Varios
# ==============================================================================
import multiprocessing
import os
import sys
from datetime import datetime
from typing import List, Dict, Iterable, Union

import logging

applic_logger = logging.getLogger('applicLogger')
data_logger = logging.getLogger('dataLogger')
train_logger = logging.getLogger('trainLogger')

## Display all the columns of the dataframe
pd.pandas.set_option('display.max_columns', None)

DATA_FOLDER = '../data'
# OUTPUT_FOLDER = '../output'
MODELS_FOLDER = '../models'
TRAIN_RESULTS_FILE = 'train_results.csv'
# UTILS_FOLDER = '../utils'
# EDA_DATAFILE = 'life_expectance_EDA.csv'


NUM_CORES = multiprocessing.cpu_count() - 1
TARGET = 'Life_expectancy'


def train_valid_xgboost_split(X_train: Union[pd.DataFrame, np.array], y_train: Union[pd.Series, np.array],
                              valid_size: float = 0.1):
    np.random.seed(123)
    idx_valid = np.random.choice(X_train.shape[0], size=int(X_train.shape[0] * valid_size), replace=False)

    X_val = X_train.iloc[idx_valid, :].copy()
    y_val = y_train.iloc[idx_valid].copy()

    X_train_valid = X_train.reset_index(drop=True).drop(idx_valid, axis=0).copy()
    y_train_valid = y_train.reset_index(drop=True).drop(idx_valid, axis=0).copy()

    return X_train_valid, X_val, y_train_valid, y_val



# Modelos a entrenar
# ==============================================================================
metric_eval = 'neg_root_mean_squared_error'
dict_models = dict()

dict_models.update({'Lasso': Lasso()})
dict_models.update({'GBRegressor': GradientBoostingRegressor(n_estimators=500, random_state=22, verbose=0,
                                                             validation_fraction=0.1, n_iter_no_change=5, tol=1.e-4)})

dict_models.update({'XGBRegressor': XGBRegressor(n_estimators=1000, eval_metric='rmse', objective='reg:squarederror')})

dict_params_models_grid = dict()

dict_params_models_grid.update({'Lasso': {'alpha': np.logspace(-3, 3, 7)}})

dict_params_models_grid.update({'GBRegressor': {'learning_rate': np.logspace(-3, 0, 4), 'max_depth': [3, 5, 7, 10],
                                                'subsample': [0.5, 0.7, 1.0],
                                                'max_features': [0.50, 'sqrt', None]}})

dict_params_models_grid.update({'XGBRegressor': {'learning_rate': [0.001, 0.01, 0.1],
                                                 'max_depth': [5, 7, 10],
                                                 'min_child_weight': [1, 3, 5],
                                                 'subsample': [0.5, 0.7, 1],
                                                 'colsample_bytree': [0.5, 0.7, 1]}})



# Modelo base
# ==============================================================================
base_model = LinearRegression(n_jobs=NUM_CORES)



def create_results_df(models_results: List[Dict]) -> pd.DataFrame:
    df_results = pd.DataFrame(columns=models_results[0].keys())

    for modelo in models_results:
        df_results = df_results.append(pd.Series(modelo), ignore_index=True)

    return df_results.sort_values(by=df_results.columns.to_list()[-1], ascending=False)


def save_models(models_results: pd.DataFrame, n_best_models: int = 1):
    for n in range(n_best_models):
        name = models_results.iloc[n, 0]
        filename = '_'.join([name, str(n + 1) + '.pkl'])
        filepath = os.path.join(MODELS_FOLDER, filename)

        model = models_results.iloc[n, 1]

        joblib.dump(model, filename=filepath)

        head_print = '<BEST>' if n == 0 else ''
        train_logger.info(f'{head_print} file: {filename}')

    train_logger.info(f'Guardado(s) {n_best_models} modelo(s).')
    applic_logger.info(f'Guardado(s) {n_best_models} mejor(es) modelo(s).')



def train_baseline_model(Xprx_train, Xprx_test, y_train, y_test, baseline_model):

    baseline_model.fit(Xprx_train, y_train)

    train_logger.info(f'**** BASELINE MODEL: {baseline_model} ****')
    train_logger.info(f'TRAIN, R cuadr: {baseline_model.score(Xprx_train, y_train)}')
    train_logger.info(f'TEST, R cuadr: {baseline_model.score(Xprx_test, y_test)}')

    baseline_metrics = cross_val_score(baseline_model, Xprx_train, y_train, scoring=metric_eval)

    train_logger.info('Generalización (cross val train), RMSE:')
    train_logger.info(f'mean: {np.abs(baseline_metrics.mean())}, std: {baseline_metrics.std()}')

    y_pred = baseline_model.predict(Xprx_test)

    train_logger.info(f'RMSE TEST: {mean_squared_error(y_test, y_pred, squared=False)}')


def train(Xprx_train: pd.DataFrame, Xprx_test: pd.DataFrame, y_train: Union[pd.Series, np.array],
          y_test: Union[pd.Series, np.array]) -> pd.DataFrame:

    applic_logger.info('INICIO TRAIN MODELS')
    applic_logger.info(f'MODELS: {list(dict_models.keys())}')
    train_logger.info('============== INICIO TRAIN MODELS ==============')

    # BASELINE MODEL
    train_baseline_model(Xprx_test, Xprx_train, y_test, y_train, baseline_model=base_model)

    # grid search: results
    results_list = list()

    for label_model, model in dict_models.items():

        param_grid = dict_params_models_grid[label_model]

        train_logger.info(f'**** INICIO MODELO {label_model} ****')
        loginfo_params = ''
        for param, values in param_grid.items():
            loginfo_params += f'<{param}>: {values}; '
        train_logger.info(f'grid params: {loginfo_params}')

        grid_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=metric_eval,
                                  n_jobs=NUM_CORES, cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=22),
                                  verbose=0, return_train_score=True)

        inic_time = datetime.now()

        if label_model == 'XGBRegressor':

            Xtrainvalid, Xval, ytrainvalid, yval = train_valid_xgboost_split(Xprx_train, y_train, valid_size=0.1)

            fit_params_xgb = {"early_stopping_rounds": 5,
                              "eval_metric": "rmse",
                              "eval_set": [(Xval.values, yval.values)],
                              "verbose": False}

            _ = grid_model.fit(Xtrainvalid.values, ytrainvalid.values, **fit_params_xgb)

        else:

            _ = grid_model.fit(Xprx_train.values, y_train.values)

        train_model_time = str(datetime.now() - inic_time)

        train_logger.info(f'tiempo: {train_model_time}')
        train_logger.info(f'Best params: {grid_model.best_params_}')

        score_train = grid_model.score(Xprx_train.values, y_train.values)
        score_test = grid_model.score(Xprx_test.values, y_test.values)

        train_logger.info(f'TRAIN {metric_eval}: {score_train}')
        train_logger.info(f'TEST {metric_eval}: {score_test}')

        result = {'label': label_model, 'model': grid_model.best_estimator_, 'params': grid_model.best_params_,
                  'time': train_model_time, 'train_score': score_train, 'test_score': score_test}

        results_list.append(result)

    df_results = create_results_df(results_list)
    df_results.drop(columns=['model']).to_csv(os.path.join(MODELS_FOLDER, TRAIN_RESULTS_FILE), index=False)

    applic_logger.info(f'Guardados resultados training: <{TRAIN_RESULTS_FILE}>')
    applic_logger.info('FIN TRAIN MODELS')
    train_logger.info('============== FIN TRAIN MODELS ==============')

    return df_results
