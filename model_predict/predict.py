import pandas as pd
import joblib


from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb

from ETL.etl_config import *
from helper.help_data import load_data


def load_data_for_predict():
    path = PATH_DATASET + 'dataset_test.csv'
    df_predict = load_data(path, sep=';')
    assert isinstance(df_predict, pd.DataFrame)

    X = df_predict.drop(['user_id'], axis=1)
    return X


def load_model(path_to_load=FILE_MODEL):
    print(path_to_load)
    try:
        model = joblib.load(path_to_load)
    except Exception as ex:
        raise Exception('Ошибка при загрузке модели:'+path_to_load, ex)

    print('Модель успешно загружена!')
    return model


def scale_predict(X_pr):
    m_scaler = load_model(FILE_SCALER)
    X_predict = m_scaler.transform(X_pr)
    return X_predict


def predict_model(X_predict):
    clf = load_model(FILE_MODEL)
    print(type(clf))
    y_pred = clf.predict_proba(X_predict)
    print(y_pred.shape)
    #clf.    .pre(X_train, y_train, eval_metric='aucpr', verbose=True)

    #return clf


def main():
    X_pr = load_data_for_predict()
    X_pr_scaled = scale_predict(X_pr)
    model = predict_model(X_pr_scaled)
    #save_model(model=model)


if __name__ == '__main__':
    main()