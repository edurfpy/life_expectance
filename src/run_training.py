#
#
#

import os
from sklearn.model_selection import train_test_split
from src.load_data import etl_data
from src.custom_preprox.custom_transf_preprox import CustomLexpectPreprx

from src.train import train, save_models

import logging
import logging.config

logging.config.fileConfig(fname='logging.conf', disable_existing_loggers=False)

applic_logger = logging.getLogger('applicLogger')
data_logger = logging.getLogger('dataLogger')

# ==============================================================================
# CONFIGURACION

DATA_FOLDER = '../data'
DATA_FILENAME = 'Life Expectancy Data.csv'
DATAFILE_PATH = os.path.join(DATA_FOLDER, DATA_FILENAME)

MODEL_FOLDER = '../models'
TRAIN_RESULTS_FILENAME = 'train_results.csv'
TRAIN_RESULTS_PATH = os.path.join(MODEL_FOLDER, TRAIN_RESULTS_FILENAME)

TARGET = 'Life_expectancy'

test_size = 0.2
preprx_pwr = True
preprx_ret_df = True


# ==============================================================================

applic_logger.info('============== INICIO RUN_TRAINING ==============')

# carga y primeras operaciones basicas

df_le = etl_data(origin=DATAFILE_PATH)


# split estratificado según 'Status' por su influencia sobre el target
X_train, X_test, y_train, y_test = train_test_split(df_le.drop(columns=TARGET), df_le[TARGET], test_size=test_size,
                                                    random_state=12, stratify=df_le.Status)

data_logger.debug(f'TRAIN_TEST_SPLIT: con test size {test_size}: \tTRAIN {len(X_train)} obsv.\tTEST {len(X_test)} obsv')


# PREPROCESADO

applic_logger.info(f'INICIO PREPROCESADO con <power transform>: {preprx_pwr}')
data_logger.info('============== INICIO PREPRX ==============')


preprx = CustomLexpectPreprx(pwr_transf=preprx_pwr, ret_df=preprx_ret_df)

X_train_preprx = preprx.fit_transform(X_train)
X_test_preprx = preprx.transform(X_test)

log_prx_mean_train = X_train_preprx.select_dtypes(include='number').mean().tolist()
log_prx_mean_test = X_test_preprx.select_dtypes(include='number').mean().tolist()
log_prx_std_train = X_train_preprx.select_dtypes(include='number').std().tolist()
log_prx_std_test = X_test_preprx.select_dtypes(include='number').std().tolist()

data_logger.debug(f'PREPROCESADO, Con <power transform> {preprx_pwr}: ')
data_logger.debug(f'Medias: \tTRAIN {log_prx_mean_train};\tTEST {log_prx_mean_test}')
data_logger.debug(f'Desv. tip.: \tTRAIN {log_prx_std_train};\tTEST {log_prx_std_test}')

applic_logger.info('FIN PREPROCESADO')
data_logger.info('============== FIN PREPRX ==============')


# TRAIN MODELS

train_results = train(X_train_preprx, X_test_preprx, y_train, y_test)

n_models = int(input('# best models to save?: '))

save_models(models_results=train_results, n_best_models=n_models)

applic_logger.info('============== FIN RUN_TRAINING ==============')