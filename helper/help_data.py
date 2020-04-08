import pandas as pd

def load_data(path_to_load: str, **kwargs)->pd.DataFrame:
    try:
        df = pd.read_csv(path_to_load, **kwargs)
    except Exception as ex:
        raise Exception('Ошибка при загрузке файла:'+path_to_load, ex)


    return df
