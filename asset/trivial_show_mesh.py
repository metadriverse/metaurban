import sys
import trimesh
import os
import time

# mesh_path = "C:\\research\\dataset\\download_asset\\walking\\dennis_posed_004_-_male_standing_business_model\\scene.gltf"
mesh_path = "C:\\research\\gitplay\\MetaVQA\\metaurban\\assets\\models\\test\\car-76823fc352eb48208eea5633827fe0cf.glb"
with open(mesh_path, 'r') as f:
    # flag_filename = sys.argv[2]
    mesh = trimesh.load(mesh_path)
    mesh.show()

#     while os.path.exists(flag_filename):
#         time.sleep(0.5)  # Poll every 0.5 seconds
# else:
#     print("Please provide a path to a .glb file and a flag filename.")

