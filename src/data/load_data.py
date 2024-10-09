import os
import sys

import pandas as pd

project_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_path)

from src.utils.logging_utils import get_logger

def read_file(file_path: str) -> pd.DataFrame:
    logger = get_logger(__name__)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension == '.xlsx' or file_extension == '.xls':
        return pd.read_excel(file_path)
    elif file_extension == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {file_extension}")
    logger.info(f"File {file_path} read successfully")