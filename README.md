# Training Vision Transformers using the Pascal VOC 2012 dataset for multi-label classification task

## Pascal VOC 2012 dataset

The dataset contains about 11,530  labelled images facilitated for numerous vision tasks including Classifications.
There are 20 Object Categories:  'aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor'

## Training

* Vision Transformer is trained using Pytorch Library
* Pytorch library `torchvision.datasets.VOCDetection` is used as Dataset for the DataLoader
* Loss criteria : `BCEWithLogitsLoss`
* Evaluation : Mean Average Prediction of labels

| Hyperparameter | Values - Trained from Scratch | Values - Pretrained from ImageNet |
| ------------- | ------------- | ------------- |
| Input Image Size  | 224 x 224  | 224 x 224  |
| Batch size  | 20  | 20  |
| Epochs  | 50  | 15  |
| Optimizer  | SGD with momentum  | SGD with momentum  |
| Learning rate  | 1e-3 | 5e-5  |
| Scheduler  | StepLR  | StepLR  |
| Momentum  | 0.9  | 0.9  |

## Performance



