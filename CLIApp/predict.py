import torch
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from CLIApp.utilities import get_input_args


def get_cat_names(category_names_file):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(abs_path, category_names_file)
    try:
        with open(full_path, 'r') as FILE:
            names = json.load(FILE)
    except FileNotFoundError as err:
        print(f'Could not find {category_names_file}')
        return None
    
    return {int(key): value for key, value in names.items()}

def load_checkpoint(filepath, device):
    # Load the checkpoint
    print('Loading model...', end='')
    try:
        abs_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(abs_path, filepath)
        checkpoint = torch.load(full_path, weights_only=False, map_location=torch.device(device))
    
        # Get base model
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
    print('Process image...', end='')
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
    print('Peforming inference...', end='')
    image = image.unsqueeze(0)
    model.to(device)
    image.to(device)

    model.eval()
    with torch.no_grad():
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(top_k, dim=1)
        
        # Move results back to CPU before converting to numpy
        top_ps = top_ps.detach().cpu().numpy().tolist()[0]
        top_class = top_class.detach().cpu().numpy()[0]
        
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_labels = [idx_to_class[lab] for lab in top_class]
        
    print(' done!\n')
    return top_ps, top_labels

def display_result(image, cat_to_name, results):
    # stdout output
    print(f'Predicted probabilities: {results[0]}')
    print(f'Predicted labels: {results[1]}')
    predicted_index = np.argmax(results[0])
    predicted_label = int(results[1][predicted_index])
    print(f'predicted flower is {cat_to_name[predicted_label]} ',
                                 f'prob = {results[0][predicted_index] * 100:.2f}%')
    
    # nice output
    _, (ax1, ax2) = plt.subplots(ncols=2)

    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax1.axis('off')
    ax1.imshow(image)

    ax2.barh(np.arange(len(results[1])), results[0])
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(results[1])))
    ax2.set_yticklabels([cat_to_name[int(x)] for x in results[1]], size='small')

    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()


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
    device = 'cpu'
    if args.gpu:
        # Verify GPU is available and enabled
        if not torch.cuda.is_available():
            print("GPU is not detected on this platform, or is not enabled.")
            response = input("Proceed on CPU? (yes, no)")
            if response.lower() == 'no' or response.lower() == 'n':
                exit()
        else:
            device = 'cuda'
            
    # Load model checkpoint
    model = load_checkpoint(
        os.path.join(args.checkpoint_path, args.model_checkpoint), 
        device
    )
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
    display_result(processed_image, names, results)
    
    
        
    
            
    
            
    
            
    
        
    
    
    
    