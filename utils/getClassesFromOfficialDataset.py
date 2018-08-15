import os
import sys
import pickle
meta = {
    'filename_cifar10': 'batches.meta',
    'filename_cifar100': 'meta',
    'key_cifar10': 'label_names',
    'key_cifar100': 'fine_label_names'
}
def cifar_load_meta(root, base_folder, name='cifar10'):
    path = os.path.join(root, base_folder, meta['filename_%s' % name])
    with open(path, 'rb') as infile:
        if sys.version_info[0] == 2:
            data = pickle.load(infile)
        else:
            data = pickle.load(infile, encoding='latin1')

        classes = data[meta['key_%s' % name]]
    return classes
