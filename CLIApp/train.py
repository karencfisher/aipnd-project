import torch
from torch import nn
from torchvision import models, datasets, transforms
from collections import OrderedDict
import os

from CLIApp.utilities import get_input_args 


def build_data_loaders(data_dir):
    '''
    Build the data loaders
    
    :param data_dir: data directory
    
    returns dictionary of data loaders
    '''
    data_loaders = {}
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    try:
        image_datasets_train = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=train_transforms
        )
        image_datasets_valid = datasets.ImageFolder(
            os.path.join(data_dir, 'valid'),
            transform=valid_transforms
        )
    except FileExistsError:
        print("Invalid data directory")
        return None
    
    data_loaders['train'] = torch.utils.data.DataLoader(
        image_datasets_train,
        batch_size=64,
        shuffle=True
    )
    data_loaders['valid'] = torch.utils.data.DataLoader(
        image_datasets_valid,
        batch_size=64
    )
        
    return data_loaders
    
def get_base_model(model_name):
    '''
    Load the base model from Torchvision
    
    :param model_name: model name
    
    returns the model
    '''
    try:
        model = getattr(models, model_name)(weights='DEFAULT')
    except AttributeError as err:
        print(f'{model_name} is not a valid Torchvision model')
        return None
    return model

def get_classifier_input_features(model):
    '''
    Gets the input dimension for the classifier
    
    :param model: The Torchvision model to be used
    
    returns the input dimension
    '''
    if hasattr(model, 'classifier'):
        return model.classifier[0].in_features, model.classifier
    elif hasattr(model, 'fc'):
        return model.fc.in_features, model.fc
    else:
        print(f"Specified model does not have a recognized classifier structure.")
        return None
    
def build_classifier(input_features, hidden_units, output_features=102):
    '''
    Builds a sequential classifier for arbitrary number of hidden layers
    
    :param input_features: input dimension 
    :param hidden_units: units for each hidden layer (list)
    :param output_features: number of output features
    
    returns Sequential object
    '''
    layers = []
       
    # Construct hidden layers
    input_dim = input_features
    for i, hidden_unit in enumerate(hidden_units):
        layers.append((f'h{i+1}', nn.Linear(input_dim, hidden_unit)))
        input_dim = hidden_unit
        layers.append(('relu', nn.ReLU()))
    
    # append output layer
    layers.append(('output', nn.Linear(input_dim, output_features)))
    layers.append(('softmax', nn.LogSoftmax(dim=1)))
    
    return nn.Sequential(OrderedDict(layers))
    
def train_model(model, data_loaders, learning_rate, epochs, gpu=True):
    '''
    Train the model, logging results
    
    :param model: the model to be trained
    :param data_loader: dictionary of data loaders (train and validation)
    :param learning_rate: learning rate (float)
    :param epochs: number of epochs (int)
    
    returns trained model
    '''
    pass

def validate_model(model, data_loader):
    '''
    Test the model, eithr on test or validation
    
    :param model: the model to test
    :param data_loader: the valid dataloader
    
    returns loss, accuracy
    '''
    return (0, 0)


if __name__ == "__main__":
    args = get_input_args('train_args.json')
    
    data_loaders = build_data_loaders(args.data_directory)
    if data_loaders is None:
        print('Failed building data loaders')
        exit()
    
    base_model = get_base_model(args.arch)
    if base_model is None:
        print('Failed loading base model')
        exit()
        
    input_features, clf = get_classifier_input_features(base_model)
    if input_features is None:
        print("Failed getting classifier input dimensions")
        exit()
        
    clf = build_classifier(input_features, args.hidden_units)
    
    trained_model = train_model(base_model, data_loaders, args.learning_rate, args.epochs, args.gpu)
    trained_model.class_to_idx = data_loaders['train'].class_to_idx
    torch.save(trained_model.state_dict(), f'./models/.{args.arch}.pth')
    
    
    
    
    