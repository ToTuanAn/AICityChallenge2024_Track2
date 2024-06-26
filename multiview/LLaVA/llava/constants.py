CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

#########################################
# Video-Llava
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<im_patch>" # ?
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
VIDEO_PLACEHOLDER = "<video-placeholder>"

MAX_IMAGE_LENGTH = 16
MAX_VIDEO_LENGTH = 1 # only 1?
PAD_LENGTH = 620

NUM_CAMERA_VIEWS = 5
NUM_PATCHES_POOLED = 16
HIDDEN_SIZE_POOLED = 256

MV_NUM_HEADS = 4
MV_DROPOUT = 0.1
#########################################