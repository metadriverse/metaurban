"""
This script go over annotated 3D models from GLB files and generates images based.
It reads configuration settings to define the paths for the JSON and GLB directories.
For each annotated JSON file, it extracts model information and filters specific types.
It then loads the corresponding GLB file,
checks if it's a Scene or a Mesh, and takes a screenshot of the loaded 3D model.
The screenshot is saved in a specified output folder with a constructed filename based on
the model's general and detail type information.
The script is particularly designed to handle special cases such as car models differently.
"""
import json
import os
import trimesh
from asset.read_config import configReader
from trimesh.viewer import SceneViewer
# Read the configuration settings
config = configReader()
path_config = config.loadPath()

# Define the paths for the JSON and GLB directories
json_folder_path = path_config["adj_parameter_folder"]
glb_folder_path = 'E:\\Projects\\metavqa_main_branch\\metaurban\\assets\\test'
output_folder_path = 'E:\\Projects\\metavqa_main_branch\\metaurban\\assets\\test_image'

# Ensure the output folder exists
os.makedirs(output_folder_path, exist_ok=True)

# Iterate over each JSON file in the folder
for json_file in os.listdir(json_folder_path):
    if json_file.endswith('.json'):
        json_path = os.path.join(json_folder_path, json_file)

        # Read the JSON file and extract relevant data
        with open(json_path, 'r') as file:
            data = json.load(file)

        # Handle specific file names and extract model information
        if json_file.startswith('car_'):
            # Special handling for car files - currently continues to next iteration
            # Use MODEL_PATH and remove 'test/' prefix
            # model_path = data['MODEL_PATH']
            # glb_filename = model_path.replace('test/', '')
            continue
        else:
            # Regular file handling - extracts general and detail type information
            glb_filename = data['filename']
            general_type = data['general']['general_type']
            detail_type = data['general']['detail_type']
        # if detail_type != "Wheelchair" and detail_type != "Scooter":
        #     continue
        # Construct the path for the GLB file
        glb_path = os.path.join(glb_folder_path, glb_filename)

        # Load the GLB file
        loaded = trimesh.load(glb_path)

        # Load the GLB file and check if it's a Scene or a Mesh
        if isinstance(loaded, trimesh.Scene):
            scene = loaded
        else:
            # Create a Scene if a single Mesh is loaded
            scene = trimesh.Scene(loaded)

        # Take a screenshot
        png = scene.save_image(resolution=[1920, 1080], visible=True)

        #  Construct the output filename and save the screenshot
        output_filename = f"{general_type}-{detail_type}-{glb_filename}.jpg"
        output_path = os.path.join(output_folder_path, output_filename)

        # Save the screenshot
        with open(output_path, 'wb') as file:
            file.write(png)

print("Processing complete.")