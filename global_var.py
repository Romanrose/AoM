import sys, os

## local paths
root_dir = './'  # root dir
src_dir = root_dir + '/src/'  # src dir
config_dir = root_dir + '/configs/'  # config dir
pretrain_dir = root_dir + '/pretrain/'  # pretrain dir

## data paths

data_info_dir = '/home/ljy/data/media/datasets/IJCAI2019_data/jsons'  # data info path
checkpoint_base = '/home/ljy/data/media/projects/aom/checkpoints'  # checkpoint base dir
log_dir = root_dir + '/logs/'  # log dir
result_dir = root_dir + '/results/'  # result dir
record_dir = root_dir + '/records/'  # record dir
checkpoint_dir = root_dir + '/checkpoints/'  # checkpoint dir

## AoM specific paths
TRC_ckpt_dir = checkpoint_dir + '/pytorch_model.bin'  # TRC pre-trained model
AoM_ckpt_dir = checkpoint_dir + '/AoM-ckpt/'  # AoM checkpoint directory
MAESC_ckpt_dir = checkpoint_dir + '/pytorch_model.bin'  # MAESC checkpoint

## Unified training checkpoint paths
train15_ckpt_dir = checkpoint_dir + 'train_15/'
train17_ckpt_dir = checkpoint_dir + 'train_17/'
train_trc_ckpt_dir = checkpoint_dir + 'train_trc/'

## Unified training log paths
twitter15_log_dir = log_dir + 'aom_twitter15_logs/'
twitter17_log_dir = log_dir + 'aom_twitter17_logs/'

## Datasets path
twitter15_data_dir = '/home/ljy/data/media/projects/aom/src/data/twitter2015/'
twitter17_data_dir = '/home/ljy/data/media/projects/aom/src/data/twitter2017/'
TRC_data_dir = '/home/ljy/data/media/projects/aom/src/data/TRC/'

## Sentiment knowledge
senticnet_path = '/home/ljy/data/media/projects/aom/src/senticnet_word.txt'

## Image paths
twitter15_img_dir = '/home/ljy/data/media/datasets/IJCAI2019_data/IJCAI2019_data/twitter2015_images'
twitter17_img_dir = '/home/ljy/data/media/datasets/IJCAI2019_data/IJCAI2019_data/twitter2017_images'

## Model paths
bart_model_dir = '/home/ljy/data/media/projects/aom/plm/bart-base'  # bart model dir
resnet152_path = '/home/ljy/data/media/projects/aom/plm/resnet/resnet152-b121ed2d.pth'
resnet152_alt_path = '/home/ljy/data/media/projects/aom/plm/resnet/resnet152-b121ed2d.pth'

## JSON config paths
twitter15_info_path = src_dir + 'data/jsons/twitter15_info.json'
twitter17_info_path = src_dir + 'data/jsons/twitter17_info.json'
trc_info_path = src_dir + 'data/jsons/TRC_info.json'

## Pretrained model paths (from MAESC_training.py)
trc_pretrain_default = 'checkpoints/pytorch_model.bin'
trained_file_default = '/home/zhouru/ABSA4/train17/2022-11-23-16-49-35/pytorch_model.bin'
senti_pretrain_default = '/home/zhouru/ABSA4/checkpoint_dir/2022-11-30-11-04-51/model45_minloss/pytorch_model.bin'

## AoM checkpoint paths
aom_twitter15_ckpt = AoM_ckpt_dir + 'Twitter2015/AoM.pt'
aom_twitter17_ckpt = AoM_ckpt_dir + 'Twitter2017/AoM.pt'


def global_update(args):
    """Update global variables based on args"""
    # Ensure directories exist
    all_dirs = [log_dir, result_dir, record_dir, checkpoint_dir,
                train15_ckpt_dir, train17_ckpt_dir, train_trc_ckpt_dir,
                twitter15_log_dir, twitter17_log_dir]
    for dir_path in all_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    return args
