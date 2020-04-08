import pandas as pd
import joblib

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb

from ETL.etl_config import *
from helper.help_data import load_data


def load_data_for_train():
    path = PATH_DATASET + 'dataset_train.csv'
    df_train = load_data(path, sep=';')
    assert isinstance(df_train, pd.DataFrame)

    X = df_train.drop(['user_id', 'is_churned'], axis=1)
    y = df_train['is_churned']
    return X, y


def save_model(model: object, path_to_save=FILE_MODEL):
    print(path_to_save)
    try:
        joblib.dump(model, path_to_save)
    except Exception as ex:
        raise Exception('Ошибка при записи модели:'+path_to_save, ex)

    print('Модель успешно сохранена!')


def scale_balance_train(X_tr, y_tr):
    m_scaler = MinMaxScaler()
    X_sc = m_scaler.fit_transform(X_tr)
    save_model(m_scaler, FILE_SCALER)
    X_train_balanced, y_train_balanced = SMOTE(random_state=42, sampling_strategy=0.2).fit_sample(X_sc, y_tr)
    return X_train_balanced, y_train_balanced


def train_model(X_train, y_train):
    clf = xgb.XGBClassifier(max_depth=3,
                            n_estimators=100,
                            learning_rate=0.1,
                            nthread=5,
                            subsample=1.,
                            colsample_bytree=0.5,
                            min_child_weight=3,
                            reg_alpha=0.,
                            reg_lambda=0.,
                            seed=42,
                            missing=1e10,
                            tree_method='gpu_hist',
                            n_jobs=-1)

    clf.fit(X_train, y_train, eval_metric='aucpr', verbose=True)

    return clf


def main():
    X, y = load_data_for_train()
    X_train_balanced, y_train_balanced = scale_balance_train(X, y)
    model = train_model(X_train_balanced, y_train_balanced)
    save_model(model=model)


if __name__ == '__main__':
    main()