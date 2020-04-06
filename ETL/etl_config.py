
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

PATH_RAW_DATA = '../data/'
PATH_RAW_DATA_TRAIN = PATH_RAW_DATA + 'train/'
PATH_RAW_DATA_TEST = PATH_RAW_DATA + 'test/'

PATH_DATASET = PATH_RAW_DATA + 'dataset/'


# Следует из исходных данных
CHURNED_START_DATE = '2019-09-01'
CHURNED_END_DATE = '2019-10-01'

INTER_1 = (1, 7)
INTER_2 = (8, 14)
INTER_3 = (15, 21)
INTER_4 = (22, 28)
INTER_LIST = [INTER_1, INTER_2, INTER_3, INTER_4]