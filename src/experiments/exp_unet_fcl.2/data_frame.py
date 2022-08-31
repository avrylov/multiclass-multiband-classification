import os
from typing import Dict, Tuple, Any

import pandas as pd

from settings import S2_DATA_FOLDER_PATH


def make_data_frame(csv_folder_path: str,
                    train_name: str,
                    validate_name: str,
                    test_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    train_df_path = os.path.join(csv_folder_path, train_name)
    validate_df_path = os.path.join(csv_folder_path, validate_name)
    test_df_path = os.path.join(csv_folder_path, test_name)

    train_df = pd.read_csv(train_df_path)
    validate_df = pd.read_csv(validate_df_path)
    test_df = pd.read_csv(test_df_path)

    train_df['s2_file_names'] = train_df.apply(lambda x:
                                               os.path.join(S2_DATA_FOLDER_PATH, x['s2_file_names']),
                                               axis=1)
    validate_df['s2_file_names'] = validate_df.apply(lambda x:
                                                     os.path.join(S2_DATA_FOLDER_PATH, x['s2_file_names']),
                                                     axis=1)
    test_df['s2_file_names'] = test_df.apply(lambda x:
                                             os.path.join(S2_DATA_FOLDER_PATH, x['s2_file_names']),
                                             axis=1)

    dataset = {
        'dataset': [train_df_path, validate_df_path, test_df_path]
    }

    data = {
        'train': train_df.copy(),
        'test': test_df.copy(),
        'validate': validate_df.copy()
    }
    return dataset, data
