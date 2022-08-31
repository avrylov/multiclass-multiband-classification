import os
from decouple import config

import torch


AVAIL_GPUS = min(1, torch.cuda.device_count())

PROJECT_ROOT = config('PROJECT_ROOT')  # abs path to src root
S2_DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT, 's2_data')
CSV_FOLDER_PATH = os.path.join(PROJECT_ROOT, 'csv')

EXP = 'res_net'
SUB_EXP = EXP + '.13'

LOGGER_PATH = os.path.join(PROJECT_ROOT, 'logs/scalar')
LOGGER_EXP_PATH = os.path.join(LOGGER_PATH, SUB_EXP)

CHECK_POINTER_PATH = os.path.join(PROJECT_ROOT, 'check_points')
CHECK_POINTER_EXP_PATH = os.path.join(CHECK_POINTER_PATH, SUB_EXP)

class_names = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]

