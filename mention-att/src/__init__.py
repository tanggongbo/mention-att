from .entity_translation_task import *
from .entity_translation_dataset import *
from .entity_translation_criterion import *
from .entity_translation_model import *
from .utils import *
import torch

# to deal with RuntimeError: received 0 items of ancdata
# reference link: https://www.cnblogs.com/zhengbiqing/p/10478311.html
torch.multiprocessing.set_sharing_strategy('file_system')