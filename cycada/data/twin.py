from torchvision import datasets
from .data_loader import DatasetParams
from .data_loader import register_dataset_obj, register_data_params

@register_data_params('twin')
class TWINParams(DatasetParams):
    
    num_channels = 3
    image_size   = (600,1200)
    #mean         = 0.1307
    #std          = 0.3081
    mean = 0.5
    std = 0.5
    num_cls      = 5
    target_transform = transforms.Lambda(lambda x: int(x) % 10) 
    print("====cycada/data/twin.py 15== twin여러 파라미터 가져오기============")

@register_dataset_obj('twin')
class TWIN(datasets.MNIST):
    def __init__(self, root, train=True,
            transform=None, target_transform=None, download=False):
        super(MNIST, self).__init__(root, train=train, transform=transform,
                target_transform=target_transform, download=download)

        