from easydict import EasyDict as edict
from utils.seed_utils import seed_everything

config = edict()

# ----- TRAIN SETTINGS -----
config.TRAIN = edict()
config.TRAIN.process_num = 1

config.TRAIN.batch_size = 128
config.TRAIN.validatiojn_batch_size = config.TRAIN.batch_size
config.TRAIN.accumulation_batch_size = 128
config.TRAIN.log_interval = 10
config.TRAIN.test_interval = 1
config.TRAIN.epoch = 30

config.TRAIN.init_lr = 0.0005
config.TRAIN.lr_scheduler = 'cos'

if config.TRAIN.lr_scheduler == 'ReduceLROnPlateau':
    config.TRAIN.epoch = 100
    config.TRAIN.lr_scheduler_factor = 0.1

config.TRAIN.weight_decay_factor = 1e-2
config.TRAIN.vis = False

config.TRAIN.warmup_step = 1500
config.TRAIN.opt = 'Adamw'
config.TRAIN.gradient_clip = 5

config.TRAIN.vis_mixcut = False
config.TRAIN.mix_precision = False


# ----- MODEL SETTINGS -----
config.MODEL = edict()
config.MODEL.model_path = '/kaggle/working/checkpoints/'
config.MODEL.early_stop = 30
config.MODEL.pretrained_model = None


# ----- DATA SETTINGS -----
config.DATA = edict()

# IMPORTANT: update this for Kaggle
config.DATA.data_file = '/kaggle/working/dataset.csv'

# If needed later
config.DATA.data_root_path = '/kaggle/working/utils'


# ----- SEED -----
config.SEED = 10086
seed_everything(config.SEED)

config.is_base = 1
