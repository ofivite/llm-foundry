# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from torch.optim import AdamW
from composer.optim import (
    ConstantDecayWithWarmupScheduler,
    ConstantWithWarmupScheduler,
    CosineAnnealingWithWarmupScheduler,
    DecoupledAdamW,
    LinearWithWarmupScheduler,
)

from llmfoundry.optim.adaptive_lion import DecoupledAdaLRLion, DecoupledClipLion
from llmfoundry.optim.lion import DecoupledLionW
from llmfoundry.optim.no_op import NoOp
from llmfoundry.optim.muon import Muon

from llmfoundry.optim.scheduler import InverseSquareRootWithWarmupScheduler
from llmfoundry.registry import optimizers, schedulers

optimizers.register('adalr_lion', func=DecoupledAdaLRLion)
optimizers.register('clip_lion', func=DecoupledClipLion)
optimizers.register('decoupled_lionw', func=DecoupledLionW)
optimizers.register('decoupled_adamw', func=DecoupledAdamW)
optimizers.register('adamw', func=AdamW)
optimizers.register('no_op', func=NoOp)
optimizers.register('muon', func=Muon)

schedulers.register('constant_decay_with_warmup', func=ConstantDecayWithWarmupScheduler)
schedulers.register('constant_with_warmup', func=ConstantWithWarmupScheduler)
schedulers.register(
    'cosine_with_warmup',
    func=CosineAnnealingWithWarmupScheduler,
)
schedulers.register('linear_decay_with_warmup', func=LinearWithWarmupScheduler)
schedulers.register(
    'inv_sqrt_with_warmup',
    func=InverseSquareRootWithWarmupScheduler,
)

__all__ = [
    'DecoupledLionW',
    'DecoupledClipLion',
    'DecoupledAdaLRLion',
    'NoOp',
    'Muon',
    'InverseSquareRootWithWarmupScheduler',
]
