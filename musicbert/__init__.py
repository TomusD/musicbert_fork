# isort: skip_file
from musicbert import freezable_roberta
from musicbert.token_classification import *  # isort:skip
from musicbert.token_classification_multi_task import *  # isort:skip
from musicbert.musicbert_with_extra_conditioning import *  # isort:skip

from ._musicbert import *

#(Triantafyllou) Custom patch for the trainer
from fairseq.trainer import Trainer 
from .custom_optimizer import custom_trainer
custom_trainer(Trainer)
# from .optimizer_patch import patch_trainer
# patch_trainer()