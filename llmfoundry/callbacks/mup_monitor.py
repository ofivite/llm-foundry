import logging
from typing import Any, Dict, Optional, Union

import torch

from composer.core import Callback, State
from composer.loggers import Logger

log = logging.getLogger(__name__)

class MupMonitor(Callback):
    """ <Description here>
    
    """

    def __init__(self, batch_log_interval: int = 1
    ):
        self.batch_log_interval = batch_log_interval
        
        # log.info(
        #     f'Initialized AsyncEval callback. Will generate runs at interval {interval}'
        # )
    
    def after_forward(self, state: State, logger: Logger):
        if state.timestamp.batch.value % self.batch_log_interval != 0:
            return

        logit_metrics = {}
        logit_metrics['output_logits'] = state.outputs.logits.abs().mean()

        for metric in logit_metrics:
            if isinstance(logit_metrics[metric], torch.Tensor):
                logit_metrics[metric] = logit_metrics[metric].item()

        logger.log_metrics(logit_metrics)