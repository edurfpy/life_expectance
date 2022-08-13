#
#
#

# import pandas as pd
# import numpy as np
# import multiprocessing
import os


from src.load_data import etl_data
from src.train import train, save_models

MODEL_FOLDER = '../models'
MODEL_DEPLOY = 'GBRegressor_1'


DATA = 'PENDIENTE'

# ¿¿ df = load_data(origin=DATA) ??
# opcion carga datos desde array


