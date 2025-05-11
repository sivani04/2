import os

from util import logutil

logger = logutil.logger_run

DATA_DIR = os.getenv('DATA_DIR', './filestore/')
ENV = os.getenv('ENV', 'LOCAL').upper()

if ENV == 'MTP':
    MODEL_DIR = os.getenv('OUTPUT_DIR', '')
elif ENV == 'MEP':
    MODEL_DIR = '/model-data/model/'
else:
    MODEL_DIR = './results/'

PREPROCESSED_FP = os.path.join(DATA_DIR, 'data/preprocessed.csv')

PRETRAINED_MODEL_DIR = os.path.join(DATA_DIR, 'pretrained')

MODEL_INFO_FILE = os.path.join(MODEL_DIR, 'model_info.json')
RESULT_FILE = os.path.join(MODEL_DIR, 'result.txt')
FULL_RESULT_FILE = os.path.join(MODEL_DIR, 'result.csv')

# input columns
TICKET_SUMMARY = 'Ticket Summary'
INTERACTION_CONTENT = 'Interaction content'
TEXT_COL = 'text'

# Type Columns to test
TYPE_COLS = ['y2', 'y3', 'y4']
CLASS_COL = 'full_type'
GROUPED = 'y1'
JOIN_CHAR = '^'
EMPTY_TYPE = 'none'

FORMATTED_TYPE_COLS = [f"formatted_{col}" for col in TYPE_COLS]
PRED_TYPE_COLS = [f"pred_{col}" for col in TYPE_COLS]

TRANSLATION_MODEL_NAME = 'facebook/m2m100_418M'
BERT_MODEL_NAME = 'bert-base-multilingual-cased'
SENTENCE_TRANSFORMER_MODEL_NAME = 'all-MiniLM-L6-v2'

TEST_SIZE = 0.2
SEED = 7

logger.info(f"os.env:     {os.environ}")
logger.info(f"ENV:        {ENV}")
logger.info(f"MODEL_DIR:  {MODEL_DIR}")
logger.info(f"DATA_DIR:   {DATA_DIR}")
