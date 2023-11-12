import os
import shutil

def copy_xml_files(source_folder, destination_folder):
    xml_files = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".jpg"):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, file)
                shutil.copy2(source_path, destination_path)
                xml_files.append(destination_path)
    return xml_files

source_folder = "/home/qu/Data/Dualray-new"
destination_folder = "/home/qu/Data/JDualray"

xml_files = copy_xml_files(source_folder, destination_folder)

for file in xml_files:
    print(file)

