"""
Обработка датасета на выбросы, пропуски и категориальные признаки
"""

import time
import pandas as pd

from helper.help_time import time_format
from ETL.etl_config import *


def prepare_dataset(dataset,
                    inter_list,
                    dataset_type='train',
                    dataset_path=PATH_DATASET
                    ):
    print(dataset_type)
    start_t = time.time()
    print('Dealing with missing values, outliers, categorical features...')

    # Профили
    dataset['age'] = dataset['age'].fillna(dataset['age'].median())
    dataset['gender'] = dataset['gender'].fillna(dataset['gender'].mode()[0])
    dataset.loc[~dataset['gender'].isin(['M', 'F']), 'gender'] = dataset['gender'].mode()[0]
    dataset['gender'] = dataset['gender'].map({'M': 1., 'F': 0.})
    dataset.loc[(dataset['age'] > 80) | (dataset['age'] < 7), 'age'] = round(dataset['age'].median())
    dataset.loc[dataset['days_between_fl_df'] < -1, 'days_between_fl_df'] = -1

    # Пинги
    for period in range(1, len(inter_list) + 1):
        col = 'avg_min_ping_{}'.format(period)
        dataset.loc[(dataset[col] < 0) |
                    (dataset[col].isnull()), col] = dataset.loc[dataset[col] >= 0][col].median()

    # Сессии и прочее
    dataset.fillna(0, inplace=True)
    dataset.to_csv('{}dataset_{}.csv'.format(dataset_path, dataset_type), sep=';', index=False)

    print('Dataset is successfully prepared and saved to {}, run time (dealing with bad values): {}'.
          format(dataset_path, time_format(time.time() - start_t)))


def main():
    train = pd.read_csv(PATH_DATASET + 'dataset_raw_train.csv', sep=';')
    test = pd.read_csv(PATH_DATASET + 'dataset_raw_test.csv', sep=';')

    print(train.shape, test.shape)

    prepare_dataset(dataset=train, inter_list=INTER_LIST, dataset_type='train')
    prepare_dataset(dataset=test, inter_list=INTER_LIST, dataset_type='test')


if __name__ == '__main__':
    main()
