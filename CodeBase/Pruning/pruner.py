import torch
import os
import time
from CodeBase.Pruning.measureCompressionAndAcceleration import measure_model

__all__ = ['Pruner']


class Pruner:

    def __init__(self, **kwargs):
        self.model = None
        self.original_model = kwargs.get('model')
        self.pruned_model = None
        self.cuda = kwargs.get('cuda')
        self.root = kwargs.get('root')
        self.validate_loader = kwargs.get('validate_loader')
        self.classes = kwargs.get(
            'classes', self.validate_loader.dataset.classes)
        self.criterion = kwargs.get('criterion', torch.nn.CrossEntropyLoss())
        self.input_size = torch.Size(kwargs.get(
            'input_size', (3, 32, 32)))
        self.preprocess_method = kwargs.get('preprocess_method')
        self.transfer_method = kwargs.get('transfer_method')
        self.measure_method = measure_model
        self.plugins = {}
        for name in [
                'iteration',
                'epoch',
                'batch',
                'update',
                'dataforward']:
            self.plugins.setdefault(name, [])

        for p in kwargs.get('plugins'):
            p.register(self)
            self.plugins[p.plugin_type].append(p)
        self.writer = kwargs.get('writer')
        if self.cuda:
            self.original_model.cuda()
            print("using gpu: {}".format(os.environ['CUDA_VISIBLE_DEVICES']))
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def validate(self, model):
        self.model = model
        self.model.eval()
        if self.cuda:
            self.model.cuda()
        for p in self.plugins['iteration']:
            p.reset()

        time_stamp = time.time()
        for data, label in self.validate_loader:
            if self.cuda:
                data, label = data.cuda(), label.cuda()
            with torch.no_grad():
                output, loss = self.plugins['dataforward'][0].call(
                    data, label, train=False)

            for p in self.plugins['iteration']:
                p.call(output, label, loss)

        for p in self.plugins['iteration']:
            p.logger(train=False)
        print('Validation Time: {}'.format(time.time() - time_stamp))

    def savePruningResult(self, cfg):
        pklToSave = {
            'cfg': cfg,
            'model_state_dict': self.pruned_model.state_dict()}
        savePath = os.path.join(self.root, 'pruned_model.pkl')
        torch.save(pklToSave, savePath)
        print("Saving pruned model as %s" % savePath)

    def prune(self, ratio=0.5):
        print('\nPruning Start\n')
        print("------------------------------------------")
        print("Before pre-processing the validation result is")
        self.validate(self.original_model)
        cfg, cfg_mask = self.preprocess_method(self, ratio)
        print("After pre-processing the validation result is")
        self.validate(self.original_model)
        print("------------------------------------------")
        print("Build new network and start transfer")
        self.transfer_method(self, cfg, cfg_mask)
        print("The validation result on new network is")
        self.validate(self.pruned_model)
        print("------------------------------------------")
        self.measure_method(
            self.original_model,
            self.pruned_model,
            self.input_size)
        self.savePruningResult(cfg)
