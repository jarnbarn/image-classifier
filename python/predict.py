import json
import torch
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import optim
import predict_args
from PIL import Image
import numpy as np
import torch.nn.functional as F

parser = predict_args.get_args()
args = parser.parse_args()

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
  
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
                                ('fc2', nn.Linear(1024, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    dense_classifier2 = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(64, 1024)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=.3)),
                                ('fc2', nn.Linear(1024, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))

    dense_classifier3 = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(2208, 1024)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=.3)),
                                ('fc2', nn.Linear(1024, 102)),
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
                                ('fc1', nn.Linear(9216, 4096)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=.3)),
                                ('fc2', nn.Linear(4096, 4096)),
                                ('relu2', nn.ReLU(inplace=True)),
                                ('dropout2', nn.Dropout(p=.3)),
                                ('fc3', nn.Linear(4096, 102)),
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

checkpoint = torch.load(args.trained_model)

trained_model.classifier.load_state_dict(checkpoint['classifier'])
    
if args.device == 'cuda':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #Process a PIL image for use in a PyTorch model
    
    size = 256, 256

    im = Image.open(image)  #reads in image to transform
    im.thumbnail(size)      #changes image size to be no bigger than size variable
    width, height = im.size  #defines width & height variables to be cropped
    left = int((width - 224)/2) #defines left, right, top, bottom coordinates for cropping in the center
    right = int((width + 226)/2)
    top = int((height - 224)/2)
    bottom = int((height + 224)/2)
    im = im.crop((left, top, right, bottom))
    
    np_image = np.array(im) #defines array of image


    np_image = np_image/255 #converts values to be between 0 - 1

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - mean) / std #normalizes values

    np_image = np_image.transpose(2, 0, 1) #reshapes matrix so that color dimension is first
    
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = model.to(device)
    # Implement the code to predict the class from an image file
    img = process_image(image_path)
    # convert to float tensor for model
    tense = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    tense = tense.unsqueeze_(0)
    #turn off changes to gradiant to speed up and find probabilities
    
    with torch.no_grad():
        out = model.forward(tense.to(device))
        out = F.softmax(out.data, dim = 1)

    probs, classes = out.topk(topk)
    
    probs = probs.type(torch.FloatTensor).to('cpu').numpy().flatten('F')
    
    
    classes = classes.type(torch.FloatTensor).to('cpu').numpy()
    classes = classes.astype(int)
    classes = classes.astype(str)

    
    return probs, classes

probs, classes = predict(args.image_path, trained_model, args.topk)

flower_names = [cat_to_name[n] for n in classes[0,:]]

if args.output == 'top':
    print('\n\nThe most likely class is {}, we\'re {:.2%} sure.\n\n'.format(flower_names[0], probs[0]))

elif args.output == 'topk':
    x = 1
    for i in range(len(flower_names)):
        print('The number {} most likely class is {}, with a probability of {:.2%}.'.format(x, flower_names[i], probs[i]))
        x += 1
      
