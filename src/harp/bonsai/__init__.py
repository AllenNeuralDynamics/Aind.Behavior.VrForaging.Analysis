import sys, os
import xml.etree.ElementTree as ET

// Write a class that creates an object that loads some library and adds certain paths to the system path




class Bonsai():

def load_bonsai_config(path = r"bonsai\Bonsai.config", root_path = None):


    if root_path is None:
        root_path = os.path.dirname(os.path.abspath(path))


    tree = ET.parse(path)
    root = tree.getroot()

    assembly_locations = root.findall("AssemblyLocations/AssemblyLocation")
    for i in assembly_locations:
        sys.path.insert(0,
                        os.path.join(
            root_path,
            os.path.dirname(i.attrib['location'])
            )
            )

    library_locations = root.findall("LibraryFolders/LibraryFolder")
    for i in library_locations:
        os.environ['PATH'] += os.path.join(root_path, i.attrib['path'])+";"