import zipfile, tqdm
import subprocess
zip_path = 'assets.zip'
extract_path = './'
password = input('Code given in the form:')
command = f"unzip -P {password} {zip_path} -d {extract_path}"

# Run the command
try:
    subprocess.run(command, shell=True, check=True)
    print("Files extracted successfully!")
except subprocess.CalledProcessError as e:
    print(f"Extraction failed: {e}")
    # for file_name in tqdm.tqdm(zip_ref.namelist()):
    #     try:
    #         # Try extracting each file individually
    #         zip_ref.extract(member=file_name, path='./', pwd=password)
    #     except RuntimeError as e:
    #         print(f"Failed to extract {file_name}: {e}")