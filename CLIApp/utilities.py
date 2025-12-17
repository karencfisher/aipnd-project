import argparse
import json
import os


def get_input_args(arg_template):
    '''
    Get and parse CL arguments
    
    :param arg_template: Description
    
    Returns parses arguments
    '''
    # get template
    arguments = get_template(arg_template)

    parser = argparse.ArgumentParser(exit_on_error=False)
    for argument in arguments:
        if argument["flag"] is None:
            parser.add_argument(
                argument["name"],
                type=eval(argument["type"]),
                help=argument["help"],
                default=argument["default"]
            )
        else: 
            parser.add_argument(
                argument["flag"],
                type=eval(argument["type"]),
                help=argument["help"],
                default=argument["default"]
            )
            
    try:
        results = parser.parse_args()
    except argparse.ArgumentError as err:
        print(f'Argument error: {err}')
        return None
    return results
       
# Helper function
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
   