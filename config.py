import os

cur_path = os.path.abspath(os.path.dirname(__file__))
PROJECT_NAME = os.path.split(cur_path)[1]

# train：tr validation：cv test：tt
# TYPE = 'tr'
# train
# TODO
LR = 1e-3
EPOCH = 100
TRAIN_BATCH_SIZE = 4
# TRAIN_DATA_PATH = '/data/yangyang/data/SPEECH_ENHANCE_DATA/tr/'
TRAIN_DATA_PATH = '/data/yangyang/data/Data/train/'

# validation
# TODO
VALIDATION_BATCH_SIZE = 4
# VALIDATION_DATA_PATH = '/data/yangyang/data/SPEECH_ENHANCE_DATA/cv/'
VALIDATION_DATA_PATH = '/data/yangyang/data/Data/validation/'
VALIDATION_DATA_NUM = 1000

# test
# TODO
TEST_DATA_PATH = '/data/yangyang/data/SPEECH_ENHANCE_DATA/tt/'
TEST_DATA_NUM = 10000

# model
# TODO
MODEL_STORE = os.path.join('/data/yangyang/result/module_store/', PROJECT_NAME + '/')
if not os.path.exists(MODEL_STORE):
    os.mkdir(MODEL_STORE)
    print('Create model store file  successful!\n'
          'Path: \"{}\"'.format(MODEL_STORE))
else:
    print('The model store path: {}'.format(MODEL_STORE))

# log
# TODO
LOG_STORE = os.path.join('/data/yangyang/result/log_store/', PROJECT_NAME + '/')
if not os.path.exists(LOG_STORE):
    os.mkdir(LOG_STORE)
    print('Create log store file  successful!\n'
          'Path: \"{}\"'.format(LOG_STORE))
else:
    print('The log store path: {}'.format(LOG_STORE))


# result
# TODO
RESULT_STORE = os.path.join('/data/yangyang/result/result/', PROJECT_NAME + '/')
if not os.path.exists(RESULT_STORE):
    os.mkdir(RESULT_STORE)
    print('Create validation result store file  successful!\n'
          'Path: \"{}\"'.format(RESULT_STORE))
else:
    print('The validation result store path: {}'.format(RESULT_STORE))

# other variable
# TODO
FILTER_LENGTH = 400
HOP_LENGTH = 160
EPSILON = 1e-8
NUM_WORKERS = 8
CUDA_ID = ['cuda:0']
# A stream和P stream的通道
C_A = 96
C_A_half = 48
C_P = 48
C_P_half = 24
# 文章第5页定义Cr为5
Cr = 5
IS_LOG = False
