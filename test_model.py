# Copyright (c) SenseTime.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from timm.models import create_model
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

import models_mixmim
# import models_mixmim_ft

input_size = 224
model_name = 'mixmim_base'
model = create_model(model_name)
print(f'test flops of model {model_name}')
model.eval()
flops = FlopCountAnalysis(model, torch.rand(1, 3, input_size, input_size))
print(flop_count_table(flops, max_depth=2))