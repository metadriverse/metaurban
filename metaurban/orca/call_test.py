import os
import subprocess

# os.system("")

xml_path = "task_examples_demo/custom_road2.xml"
agent_num = "2"
# ./build/single_test ../task_examples_demo/custom_road2.xml 2
command = ["./build/single_test", xml_path, agent_num]
output = subprocess.run(command, capture_output=True, text=True)

print(output.stdout)