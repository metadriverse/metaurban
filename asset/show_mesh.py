import sys
import trimesh
import os
import time

if len(sys.argv) > 1:
    mesh_path = sys.argv[1]
    # flag_filename = sys.argv[2]
    mesh = trimesh.load(mesh_path)
    mesh.show()

#     while os.path.exists(flag_filename):
#         time.sleep(0.5)  # Poll every 0.5 seconds
# else:
#     print("Please provide a path to a .glb file and a flag filename.")

