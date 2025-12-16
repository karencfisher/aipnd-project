import argparse
import json
import os


def get_input_args(arg_template):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    arg_template_path = os.path.join(
        script_dir, 
        'arg-templates', 
        arg_template
    )
    with open(arg_template_path, 'r') as FILE:
        arguments = json.load(FILE) 

    parser = argparse.ArgumentParser()
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
    return parser.parse_args()   
   

if __name__ == "__main__":
    args = get_input_args('predict_args.json')
    print(args)