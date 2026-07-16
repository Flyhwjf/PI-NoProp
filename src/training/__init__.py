from .local_trainer import LocalNoPropTrainer, configure_torch
from .pretrain import (pretrain_encoder, pretrain_decoder,
                       align_label_embeddings_to_encoder,
                       save_shared_components, load_shared_components)

# Legacy helpers have optional notebook-era dependencies.  Optimized training
# must remain importable in a minimal PyTorch environment.
try:
    from .trainer import Trainer
except ImportError:
    Trainer = None
try:
    from .metrics import evaluate
except ImportError:
    evaluate = None
