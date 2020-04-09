import yaml
from pprint import pprint
from helper.logging import logger, save_logging_config


# # -*- coding: utf-8 -*-
# import yaml
# import io
#
# # Define data
# data = {
#     'a list': [
#         1,
#         42,
#         3.141,
#         1337,
#         'help',
#         u'€'
#     ],
#     'a string': 'bla',
#     'another dict': {
#         'foo': 'bar',
#         'key': 'value',
#         'the answer': 42
#     }
# }
#
# # Write YAML file
# with io.open('data.yaml', 'w', encoding='utf8') as outfile:
#     yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)
#
# # Read YAML file
# with open("data.yaml", 'r') as stream:
#     data_loaded = yaml.safe_load(stream)
#
# print(data == data_loaded)

FILE_CFG = '../config/model_cfg.yaml'


def load_config(filename=None):

    if not filename:
        filename = FILE_CFG



    try:
        with open(filename, 'r') as f:
            config = yaml.safe_load(f.read())
            pprint(config)

    except Exception as err:
        err_text = 'Ошибка при загрузке config file:' + filename
        logger.exception(err_text)
        raise Exception(err_text, err)

    logger.info(f'Config file {filename} загружен')


load_config()


PATH_RAW_DATA = '../data_1/'
PATH_RAW_DATA_TRAIN = PATH_RAW_DATA + 'train/'
PATH_RAW_DATA_TEST = PATH_RAW_DATA + 'test/'

PATH_DATASET = PATH_RAW_DATA + 'dataset/'

PATH_MODEL = PATH_RAW_DATA + 'model/'
FILE_MODEL = PATH_MODEL + 'xgb_model.pkl'
FILE_SCALER = PATH_MODEL + 'model_scaler.pkl'

# Следует из исходных данных
CHURNED_START_DATE = '2019-09-01'
CHURNED_END_DATE = '2019-10-01'

INTER_1 = (1, 7)
INTER_2 = (8, 14)
INTER_3 = (15, 21)
INTER_4 = (22, 28)
INTER_LIST = [INTER_1, INTER_2, INTER_3, INTER_4]