import torch
from torchvision import datasets
from utils.load_imglist import ImageList

class dataset_factory(object):

    
    @staticmethod 
    def get_train_loader_and_validate_loader(dataset_name, c_transform, arg_batch_size,arg_validate_batch_size, kwargs, 
                dataset_config = None, root_path='./data', fileList_path='./data'):
        low_datasets = list((s.lower() for s in datasets.__all__))
        dataset_dict = dict(list(zip(low_datasets,datasets.__all__)))
        print(low_datasets)
        # get tran_loader and validate_loader
        if dataset_name.lower() in low_datasets:
            train_loader = torch.utils.data.DataLoader(
                getattr(datasets,dataset_dict[dataset_name])(root_path, transform=c_transform), #root transforms,train[defult=true],download[defult=false]
                batch_size=arg_batch_size, shuffle=True, **kwargs
                )
            validate_loader = torch.utils.data.DataLoader(
                getattr(datasets,dataset_dict[dataset_name])(root_path, train = False, transform=c_transform),
                batch_size=arg_validate_batch_size, shuffle=False, **kwargs
                )
        else:
            train_loader = torch.utils.data.DataLoader(
                ImageList(root=root_path, fileList=fileList_path, transform=c_transform),
                batch_size=arg_batch_size, shuffle=True, **kwargs
                )
            validate_loader = torch.utils.data.DataLoader(
                ImageList(root=root_path, fileList=fileList_path, transform=c_transform), # !!!crop size delete from command parameter,write into transfrom_config.xml
                batch_size=arg_validate_batch_size, shuffle=False, **kwargs
          )
        return train_loader, validate_loader