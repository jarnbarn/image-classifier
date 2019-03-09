import test_args
import torch
import torch
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import optim



parser = test_args.get_args()
args = parser.parse_args()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([.485,.456,.406],
                                                           [.229,.224,.225])
                                      ]) 
valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([.485,.456,.406],
                                                          [.229,.224,.225])
                                     ])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([.485,.456,.406],
                                                          [.229,.224,.225])
                                     ])

data_transforms = [train_transforms, valid_transforms, test_transforms]


#Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform = data_transforms[0])

valid_data = datasets.ImageFolder(valid_dir, transform = data_transforms[1])

test_data = datasets.ImageFolder(test_dir, transform = data_transforms[2])

image_datasets = [train_data, valid_data, test_data]

#Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True)

validloader = torch.utils.data.DataLoader(image_datasets[1], batch_size=32)

testloader = torch.utils.data.DataLoader(image_datasets[2], batch_size=32)

dataloaders = [trainloader, validloader, testloader]


if args.device == 'cuda':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
    
def get_model():
    if args.vgg == 1:
        model = models.vgg11(pretrained = True)
    elif args.vgg == 2:
        model = models.vgg13(pretrained = True)
    elif args.vgg == 3:
        model = models.vgg16(pretrained = True)
    elif args.vgg == 4:
        model = models.vgg19(pretrained = True)
    
    if args.alexnet:
        model = models.alexnet(pretrained = True)

    if args.densenet == 1:
        model = models.densenet121(pretrained = True)
    elif args.densenet == 2:
        model = models.densenet169(pretrained = True)
    elif args.densenet == 3:
        model = models.densenet161(pretrained = True)
    elif args.densenet == 4:
        model = models.densenet201(pretrained = True)
 
    for param in model.parameters():
        param.requires_grad = False 

    vgg_classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, 4096)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=.3)),
                                ('fc2', nn.Linear(4096, 4096)),
                                ('relu2', nn.ReLU(inplace=True)),
                                ('dropout2', nn.Dropout(p=.3)),
                                ('fc3', nn.Linear(4096, 2048)),
                                ('relu3', nn.ReLU(inplace=True)),
                                ('dropout3', nn.Dropout(p=.3)),
                                ('fc4', nn.Linear(2048, 1024)),
                                ('relu4', nn.ReLU(inplace=True)),
                                ('dropout4', nn.Dropout(p=.3)),
                                ('fc5', nn.Linear(1024, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    dense_classifier1 = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, 1024)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=.3)),
                                ('fc2', nn.Linear(1024, 512)),
                                ('relu2', nn.ReLU(inplace=True)),
                                ('dropout2', nn.Dropout(p=.3)),
                                ('fc3', nn.Linear(512, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    dense_classifier2 = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(64, 1024)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=.3)),
                                ('fc2', nn.Linear(1024, 512)),
                                ('relu2', nn.ReLU(inplace=True)),
                                ('dropout2', nn.Dropout(p=.3)),
                                ('fc3', nn.Linear(512, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    dense_classifier3 = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(2208, 1024)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=.3)),
                                ('fc2', nn.Linear(1024, 512)),
                                ('relu2', nn.ReLU(inplace=True)),
                                ('dropout2', nn.Dropout(p=.3)),
                                ('fc3', nn.Linear(512, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    dense_classifier4 = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1920, 960)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=.3)),
                                ('fc2', nn.Linear(960, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    alexnet_classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(9216, 4608)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=.3)),
                                ('fc2', nn.Linear(4608, 2304)),
                                ('relu2', nn.ReLU(inplace=True)),
                                ('dropout2', nn.Dropout(p=.3)),
                                ('fc3', nn.Linear(2304, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))   
                    
    
    if args.vgg:
        model.classifier = vgg_classifier
    elif args.alexnet:
        model.classifier = alexnet_classifier
    elif args.densenet == 1:
        model.classifier = dense_classifier1
    elif args.densenet == 2:
        model.classifier = dense_classifier2
    elif args.densenet == 3:
        model.classifier = dense_classifier3
    elif args.densenet == 4:
        model.classifier = dense_classifier4

    return model    
    
trained_model = get_model()

checkpoint = torch.load('checkpoint.pth')

trained_model.classifier.load_state_dict(checkpoint['classifier'])

correct = 0
total = 0

trained_model = trained_model.to(device)


with torch.no_grad():
    for data in dataloaders[2]:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = trained_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))