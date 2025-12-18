import torch
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from CLIApp.utilities import get_input_args, Timer


def sniff_gpu(gpu):
    '''
    Check for GPU if requested
    
    :param gpu: Request GPU (bool)
    
    Returns device, 'cuda' or 'cpu'
    '''
    if args.gpu:
        # Verify GPU is available and enabled
        if not torch.cuda.is_available():
            while True:
                print("GPU is not detected on this platform, or it is not enabled.")
                response = input("Proceed on CPU ('yes' or 'no')? ").lower()
                if response == 'no' or response == 'n':
                    return None
                elif response == 'yes' or response == 'y':
                    break
                else:
                    print('Invalid response')
        else:
            print("Inferring on GPU")
            return 'cuda'
    print("Inferring on CPU")
    return 'cpu'
    
def get_cat_names(category_names_file):
    '''
    Retrieve the category name mapping
    
    :param category_names_file: relative file path for category name JSON
    
    Returns dictionary mapping category ids to names
    '''
    abs_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(abs_path, category_names_file)
    try:
        with open(full_path, 'r') as FILE:
            names = json.load(FILE)
    except FileNotFoundError as err:
        print(f'Could not find {category_names_file}')
        return None
    
    return {int(key): value for key, value in names.items()}

def load_checkpoint(checkpoint_file_path, device):
    '''
    Load the model checkpoint
    
    :param filepath: Relative path to the checkpoint
    :param device: Device ('cpu' or 'cuda')
    '''
    print('Loading model...', end='', flush=True)
    try:
        # load checkpoint
        abs_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(abs_path, checkpoint_file_path)
        checkpoint = torch.load(full_path, weights_only=False, map_location=torch.device(device))
    
        # get base model
        model = getattr(models, checkpoint['arch'])(weights='DEFAULT')
        for param in model.parameters():
            param.requires_grad = False
            
        # loading weights, classifier, and class indexes
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        
    except Exception as err:
        print(f'Error loading the checkpoint: {err}')
        return None
    print(' done!')
    return model

def process_image(image_file_path):
    '''
    Process the image for inference
    
    :param image_file_path: Relative path to the image file
    
    Returns processed image
    '''
    print('Process image...', end='', flush=True)
    # get the image
    try:
        abs_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(abs_path, image_file_path)
        image = Image.open(full_path)
    except Exception as err:
        print(f'Error opening image: {err}')
        return None
    
    # Resize where the shortest side is 256 pixels, keeping the aspect ratio.
    width, height = image.size
    if width < height:
        new_width = 256
        new_height = int(256 * height / width)
    else:
        new_height = 256
        new_width = int(256 * width / height)
    image = image.resize((new_width, new_height))

    # Center crop to 224x224
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Convert to numpy array and normalize to 0-1
    np_image = np.array(image) / 255

    # Normalize with ImageNet means and stds
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np_normalized = (np_image - mean) / std

    # PyTorch expects color channel to be the first dimension
    image_np_normalized = image_np_normalized.transpose((2, 0, 1))
    
    print(' done!')

    # Convert to PyTorch tensor
    return torch.from_numpy(image_np_normalized).type(torch.FloatTensor)

def run_inference(model, device, image, top_k):
    '''
    Run inference on the processed image
    
    :param model: trained model
    :param device: device ('cpu' or 'cuda')
    :param image: processed image
    :param top_k: number of top k predictions
    
    Returns top probabilities and labels
    '''
    print('Performing inference...', end='', flush=True)
    image = image.unsqueeze(0)
    model.to(device)
    image.to(device)

    model.eval()
    with torch.no_grad():
        # Run forward pass and gets results
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(top_k, dim=1)
        
        # Move results back to CPU before converting to numpy
        top_ps = top_ps.detach().cpu().numpy().tolist()[0]
        top_class = top_class.detach().cpu().numpy()[0]
        
        # map top_class to actual labels
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_labels = [idx_to_class[label] for label in top_class]
        
    print(' done!')
    return top_ps, top_labels

def display_result(cat_to_name, results, timer):
    '''
    Displays formatted results to STDOUT
    
    :param cat_to_name: dictionary mapping ids to class names
    :param results: inference results
    
    Returns None
    '''
    # print elapsed time
    print(f'Time taken: {timer()}')
    
    # print prediction
    predicted_index = np.argmax(results[0])
    predicted_label = int(results[1][predicted_index])
    print(f'\nPredicted flower is {cat_to_name[predicted_label]}')
    
    # determine spacing for tabular results
    max_spaces = max([len(name) for name in cat_to_name.values()]) + 10
    spacing = lambda x: " " * (max_spaces - len(x))
        
    # print tabular results
    print(f'\nTop {len(results[0])} predictions:')
    print(f'\nFlower name{spacing("Flower name")}Probability')
    print("-" * (max_spaces + 13))
    for result in zip(results[1], results[0]):
        name = cat_to_name[int(result[0])].title()
        spaces = max_spaces - len(name)
        print(f'{name}{" " * spaces}{result[1] * 100:.2f}%')
    print()

if __name__ == '__main__':
    # get CL arguments
    args = get_input_args('predict_args.json')
    if args is None:
        exit()
        
    # Get category names
    names = get_cat_names(args.category_names)
    if names is None:
        exit()
    
    # Check gpu argument
    device = sniff_gpu(args.gpu)
    if device is None:
        exit()
            
    # Start a timer
    timer = Timer()
    
    # Load model checkpoint
    model = load_checkpoint(args.model_checkpoint, device)
    if model is None:
        exit()
        
    # get full image path
    abs_path = os.path.dirname(os.path.abspath(__file__))
    full_image_path = os.path.join(abs_path, 'CLIApp', args.image_file_path)
    
    # get and process image from file
    processed_image = process_image(args.image_file_path)
    if processed_image is None:
        exit()
        
    # make inference on image
    results = run_inference(model, device, processed_image, args.top_k)
    if results is None:
        exit()
        
    # display results
    display_result(names, results, timer)
    
    