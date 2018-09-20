# coding=utf-8
# measureCompressionAndAcceleration.py
from CodeBase.Utils import *


def measure_model(original_model, pruned_model):
    original_model.cpu()
    pruned_model.cpu()
    print('Model before pruning:')
    original_model_params = measure_layer_param(original_model)
    original_model_flops = measure_layer_flops(original_model)

    print('Model after pruning:')
    pruned_model_params = measure_layer_param(pruned_model)
    pruned_model_flops = measure_layer_flops(pruned_model)

    print(
        'Params_reduced:{:.2f}%\nFLOPs_reduced:{:.2f}%'.format(
            (original_model_params - pruned_model_params) * 100 /
            original_model_params,
            (original_model_flops - pruned_model_flops) * 100 /
            original_model_flops))
