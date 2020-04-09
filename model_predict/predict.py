import pandas as pd
import joblib

from helper.config import *
from helper.help_data import load_data
from helper.logging import logger


def load_data_for_predict():
    path = PATH_DATASET + 'dataset_test.csv'

    df_predict = load_data(path, sep=';')
    assert isinstance(df_predict, pd.DataFrame)

    X = df_predict.drop(['user_id'], axis=1)
    return X


def load_model(path_to_load=FILE_MODEL):
    try:
        model = joblib.load(path_to_load)
    except Exception as err:
        err_text = 'Ошибка при загрузке модели:' + path_to_load
        logger.exception(err_text)
        raise Exception(err_text, err)

    logger.info(f'Модель {path_to_load} загружена')
    return model


def scale_predict(X_pr):
    m_scaler = load_model(FILE_SCALER)
    X_predict = m_scaler.transform(X_pr)
    return X_predict


def predict_model(X_predict):
    clf = load_model(FILE_MODEL)
    logger.debug(f'predict модель: {type(clf)}')
    y_pred = clf.predict_proba(X_predict)
    logger.debug(f'результат predict_proba = {y_pred.shape}')


def main():
    X_pr = load_data_for_predict()
    X_pr_scaled = scale_predict(X_pr)
    model = predict_model(X_pr_scaled)
    #save_model(model=model)


if __name__ == '__main__':
    main()