import nbformat
import os

mode = 'off'

def set_execution_mode_off(notebook_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        if 'mystnb' in nb.metadata and 'execution_mode' in nb.metadata['mystnb']:
            nb.metadata['mystnb']['execution_mode'] = mode
            print(f"Updated execution_mode to {mode} in {notebook_path}")
        else:
            print(f"No execution_mode found in {notebook_path}. Skipping...")

        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

    except Exception as e:
        print(f"Error processing {notebook_path}: {e}")

def batch_update_execution_mode(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.ipynb'):
            notebook_path = os.path.join(directory, filename)
            set_execution_mode_off(notebook_path)

directory = "./source"  
batch_update_execution_mode(directory)
