import yaml
from pprint import pprint
from helper.logging import logger, save_logging_config


FILE_CFG = '../config/model_cfg.yaml'


def load_config(filename=None):

    if not filename:
        filename = FILE_CFG

    try:
        with open(filename, 'r') as f:
            cfg = yaml.safe_load(f.read())
            #pprint(config)

    except Exception as err:
        err_text = 'Ошибка при загрузке config file:' + filename
        logger.exception(err_text)
        raise Exception(err_text, err)

    logger.info(f'Config file {filename} загружен')
    return cfg


config = load_config()

pprint(config)


PATH_RAW_DATA = config['PATH_RAW_DATA']
PATH_RAW_DATA_TRAIN = config['PATH_RAW_TRAIN']
PATH_RAW_DATA_TEST = config['PATH_RAW_TEST']

PATH_DATASET = config['PATH_DATASET']

FILE_DATASET_SEP = config['datasets']['sep']

train_dict = config['datasets']['train']
FILE_DATASET_RAW_TRAIN = train_dict['path']+train_dict['file_raw']
FILE_DATASET_MODEL_TRAIN = train_dict['path']+train_dict['file_model']

test_dict = config['datasets']['train']
FILE_DATASET_RAW_TEST = test_dict['path']+test_dict['file_raw']
FILE_DATASET_MODEL_TEST = test_dict['path']+test_dict['file_model']

PATH_MODEL = config['PATH_MODEL']
FILE_MODEL = PATH_MODEL + config['FILE_MODEL']
FILE_SCALER = PATH_MODEL + config['FILE_SCALER']

# Следует из исходных данных
CHURNED_START_DATE = config['CHURNED_START_DATE']
CHURNED_END_DATE = config['CHURNED_END_DATE']

INTER_1 = (1, 7)
INTER_2 = (8, 14)
INTER_3 = (15, 21)
INTER_4 = (22, 28)
INTER_LIST = [INTER_1, INTER_2, INTER_3, INTER_4]