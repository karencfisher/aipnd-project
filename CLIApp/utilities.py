from torch.cuda import is_available
import argparse
import json
import os
from time import time


class Timer:
    '''
    Timer -- times a process
    
    Use:
    Start timer by instantiating instance of this class, e.g.
        timer = Timer()
        
    Get elapsed time by calling the instance, e.g.
        elapsed_time = timer()
        
    Returns elapsed time as a string in the form hh:mm:ss.sss
    '''
    def __init__(self):
        self.start_time = time()
        
    def __call__(self):
        elapsed_time = time() - self.start_time
        hours = int(elapsed_time // 3600)
        elapsed_time %= 3600
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        return f'{hours:02d}:{minutes:02d}:{seconds:06.3f}'
    
def get_full_path(path):
    '''
    Change relative to absolute path
    
    :param path: relative path
    
    Returns absolute path
    '''
    abs_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(abs_path, '..', path)

def sniff_gpu(gpu):
    '''
    Check for GPU if requested
    
    :param gpu: Request GPU (bool)
    
    Returns device, 'cuda' or 'cpu'
    '''
    if gpu:
        # Verify GPU is available and enabled
        if not is_available():
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

def get_input_args(arg_template):
    '''
    Get and parse CL arguments
    
    :param arg_template: Description
    
    Returns parses arguments
    '''
    # get template
    arguments = get_template(arg_template)

    # build arguments
    parser = argparse.ArgumentParser(exit_on_error=False)
    for argument in arguments:
        # If 'type' is included, replace the string with the actual Python type
        type_str =  argument['attributes'].get('type')
        if type_str is not None:
            argument['attributes']['type'] = eval(type_str)
        
        parser.add_argument(argument['dest'], **argument['attributes'])
            
    try:
        results = parser.parse_args()
    except argparse.ArgumentError as err:
        print(f'Argument error: {err}')
        return None
    return results
       
# Helper function for get_input_args
def get_template(arg_template):
    '''
    Helper function to open arguments template
    
    :param arg_template: Description
    
    returns arguments as list of dictionaries
    '''
    template_dir = os.path.dirname(os.path.abspath(__file__))
    arg_template_path = os.path.join(
        template_dir, 
        'arg-templates', 
        arg_template
    )
    with open(arg_template_path, 'r') as FILE:
        arguments = json.load(FILE)
    return arguments
   