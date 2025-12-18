import torch
from torch import nn
from torchvision import models, datasets, transforms
from collections import OrderedDict
from tqdm import tqdm
import os

from CLIApp.utilities import get_input_args, Timer, get_full_path, sniff_gpu 


def build_data_loaders(full_data_path):
    '''
    Build the data loaders
    
    :param data_dir: data directory
    
    Returns dictionary of data loaders
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
            os.path.join(full_data_path, 'train'),
            transform=train_transforms
        )
        image_datasets_valid = datasets.ImageFolder(
            os.path.join(full_data_path, 'valid'),
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
    
    Returns the model
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
    
    Returns the input dimension and pointer to the top node
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
    
    Returns Sequential object
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
    
def train_model(model, device, data_loaders, learning_rate, epochs):
    '''
    Train the model
    
    :param model: the model to be trained
    :param data_loader: dictionary of data loaders (train and validation)
    :param learning_rate: learning rate (float)
    :param epochs: number of epochs (int)
    
    Returns None (model trained in place)
    '''
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)

    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = 0
        model.train()
        for inputs, labels in tqdm(data_loaders['train'], desc='Training'):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        valid_loss, accuracy = validate_model(model, device, data_loaders['validate'])

        train_losses.append(train_loss/len(data_loaders['train']))
        valid_losses.append(valid_loss)
        print(f'Train loss: {train_losses[-1]:.3f} - Valid loss: {valid_losses[-1]:.3f} - ',
              f'Valid accuracy: {accuracy:.3f}\n')

def validate_model(model, device, data_loader):
    '''
    Test the model, eithr on test or validation
    
    :param model: the model to test
    :param data_loader: the valid dataloader
    
    Returns loss, accuracy
    '''
    criterion = nn.NLLLoss()
    test_loss = 0
    test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            test_loss += loss.item()

            ps = torch.exp(logps)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    loss = test_loss/len(data_loader)
    accuracy = test_accuracy/len(data_loader)
    return loss, accuracy

def save_checkpoint(arch, hidden_units, model, class_to_idx, full_save_path):
    '''
    Save the checkpoint
    
    Checkpoints will be in the format of 
    
    <base_model>_n
    
    So subsequent checkpoints of same architecture can be persisted.
    
    :param arch: Name of the base model architecture (e.g., 'vgg16')
    :param hidden_units: hidden values (list of ints)
    :param model: The trained model
    :param class_to_idx: Class to id mapping
    :param full_save_path: Absolute path to checkpoints directory
    
    Returns None
    '''
    # Get previous versions (if any)
    prev_checkpoints = sorted(os.listdir(os.path.join(full_save_path, f'{arch}_*.pth')))
    
    # Construct new checkpoint file name and save path
    if len(prev_checkpoints) > 0:
        last_version_num = int(prev_checkpoints[-1].strip().split('_')[1])
        new_version = f'{arch}_{last_version_num + 1}'
    else:
        new_version = f'{arch}_1'
    new_checkpoint_path = os.path.join(full_save_path, new_version)
    
    # Cache checkpoint information
    checkpoint = {
        "arch": "vgg16",
        "classifier": model.classifier,
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "hidden_units": hidden_units
    }
    
    torch.save(checkpoint, new_checkpoint_path)


if __name__ == "__main__":
    # Get CL arguments
    args = get_input_args('train_args.json')
    if args is None:
        exit()
        
    # Get full data path and save path and verify
    full_data_path = get_full_path(args.data_directory)
    if not os.path.exists(full_data_path):
        print("Data directory is not found.")
        exit()
        
    full_save_path = get_full_path(args.save_directory)
    if not os.path.exists(full_save_path):
        print("Save directory is not found.")
        exit()
        
    # Verify GPU enabled and get device string
    device = sniff_gpu(args.gpu)
    if device is None:
        exit()
    
    # Build data loaders
    data_loaders = build_data_loaders(full_data_path)
    if data_loaders is None:
        exit()
    
    # Download base model by name e.g., 'vgg16'
    model = get_base_model(args.arch)
    if model is None:
        exit()
        
    # Get input features and top node 
    input_features, clf_node = get_classifier_input_features(model)
    if input_features is None:
        exit()
        
    clf_node = build_classifier(input_features, args.hidden_units)
    
    train_model(model, data_loaders, args.learning_rate, args.epochs, device)
    
    save_checkpoint(args.arch, args.hidden_units, model, 
                    data_loaders['train'].class_to_idx, full_save_path)
    
    
    
    
    