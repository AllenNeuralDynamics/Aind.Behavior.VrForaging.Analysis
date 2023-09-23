import sys, os
import xml.etree.ElementTree as ET

def load_bonsai_config(bonsai_path: str = "bonsai"):

    _config = os.path.join(bonsai_path, "bonsai.config")
    xmltree = ET.parse(_config)
    root = xmltree.getroot()
    assembly_locations = root.findall("AssemblyLocations/AssemblyLocation")
    for i in assembly_locations:
        sys.path.insert(0,
                        os.path.join(bonsai_path, os.path.dirname(i.attrib['location'])))

    library_locations = root.findall("LibraryFolders/LibraryFolder")
    for i in library_locations:
        os.environ['PATH'] += os.path.join(bonsai_path, i.attrib['path'])+";"

