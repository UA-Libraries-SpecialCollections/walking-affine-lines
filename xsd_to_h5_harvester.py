#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\xsd_to_h5_harvester.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu


# This script takes a list of xml files compliant with the dataObject.xsd schema and produces a HDF5 formatted
# computational dataset file. 

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


import os
import xml.etree.ElementTree as ET
import h5py
import numpy as np
from collections import defaultdict
from dataclasses import make_dataclass, is_dataclass, fields
from typing import List, Dict, Any, get_origin, get_args
import ADO_config
from backrooms import get_files_by_extension, get_save_path_with_filename, Timer

XSD_NS = {'xs': 'http://www.w3.org/2001/XMLSchema'}
timer = Timer()
timer.start()

# Pass 1: Collect all type definitions and their fields
def parse_xsd_schema_two_pass(xsd_path: str) -> Dict[str, List[tuple]]:
    tree = ET.parse(xsd_path)
    root = tree.getroot()
    type_defs = {}

    for elem in root.findall('.//xs:element', namespaces=XSD_NS):
        elem_name = elem.get('name')
        complex_type = elem.find('xs:complexType', namespaces=XSD_NS)
        if elem_name and complex_type is not None:
            sequence = complex_type.find('xs:sequence', namespaces=XSD_NS)
            if sequence is not None:
                fields = []
                for subelem in sequence.findall('xs:element', namespaces=XSD_NS):
                    subname = subelem.get('name')
                    max_occurs = subelem.get('maxOccurs')
                    is_list = max_occurs == "unbounded" or (max_occurs and int(max_occurs) > 1)
                    if subname:
                        fields.append((subname, is_list))
                type_defs[elem_name] = fields

    return type_defs

# Pass 2: Build classes with correct types including lists
def build_dataclasses_two_pass(type_defs: Dict[str, List[tuple]]) -> Dict[str, Any]:
    class_defs = {}

    # First: create empty class shells
    for type_name in type_defs:
        class_defs[type_name] = make_dataclass(type_name, [])

    # Then: fill in fields with correct class references or Any
    for type_name, fields_list in type_defs.items():
        field_defs = []
        for f, is_list in fields_list:
            if f in class_defs:
                field_type = class_defs[f]
            else:
                field_type = Any
            if is_list:
                field_type = List[field_type]
            field_defs.append((f, field_type))
        class_defs[type_name] = make_dataclass(type_name, field_defs)

    return class_defs

# Parse XML using a nested dataclass structure derived from the xml datas schema definition
def parse_xml_with_schema(xml_path: str, class_defs: Dict[str, Any], root_class_name: str):
    print("")
    print("parse_xml_with_schema")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    print(get_origin(class_defs["dataObject"]))
    print(get_args(class_defs["dataObject"]))
    
    def parse_element(elem, class_type):
        if not is_dataclass(class_type):
            print("")
            print("following is not a dataclass")
            print(class_type)
            print("")
            return elem.text.strip() if elem.text else None
                
        
        values = {}
        for field_name, field_type in class_type.__annotations__.items():
            print("")
            print("annotations")
            print(field_name)
            print(field_type)
            print("")
            matches = elem.findall(f"DOB:{field_name}", namespaces={"DOB": "http://libcontent.lib.ua.edu/digital/schemas/DOB"})
            print("matches")
            print(matches)
            if not matches:
                print(r"if not matches:1")
                matches = elem.findall(field_name)

            origin = get_origin(field_type) # <---- I changed this to matches to test for list, still need to check output stream
            args = get_args(field_type)
            print("")
            print("origin")
            print(origin)
            print("args")
            print(args)
            print("")
            
            if origin == list:
                print(r"yes")
                print(r"origin == list:")
                print(r"")
                if is_dataclass(args[0]):
                    print(r"yes")
                    print(r"is_dataclass(args[0])")
                    print(r"")

            if not matches:
                print(r"if not matches:2")
                values[field_name] = None

            elif origin == list and is_dataclass(args[0]):
                print(r"elif origin == list and is_dataclass(args[0]):")
                print(r"args[0]")
                print(args[0])
                values[field_name] = [parse_element(child, class_defs[field_name]) for child in matches]

            elif is_dataclass(field_type):
                print(r"elif is_dataclass(field_type):")
                print(r"matches[0]")
                print(matches[0])
                print(ET.tostring(matches[0], encoding='UTF-8'))
                print(field_type)
                values[field_name] = parse_element(matches[0], class_defs[field_name])
                print(r"values[field_name] = ")
                print(values[field_name])

            else:
                if origin == list:
                    print(r"if origin == list:")
                    values[field_name] = [child.text.strip() for child in matches if child.text]
                else:
                    print(r"else:")
                    print(ET.tostring(matches[0], encoding='UTF-8'))
                    values[field_name] = matches[0].text.strip() if matches[0].text else None
        print("")
        return class_type(**values)

    ns = {"DOB": "http://libcontent.lib.ua.edu/digital/schemas/DOB"}
    tag_local = root.tag.split('}')[-1]

    if tag_local == root_class_name:
        root_elem = root
    else:
        root_elem = root.find(f'.//DOB:{root_class_name}', namespaces=ns)
        if root_elem is None:
            root_elem = root.find(root_class_name)

    if root_elem is None:
        raise ValueError(f"Could not find root element '{root_class_name}' in the XML document.")


    return parse_element(root_elem, class_defs[root_class_name])

# Validate XML structure against expected dataclass types
def validate_xml_structure(xml_path: str, class_defs: Dict[str, Any], root_class_name: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    errors = []

    def validate_element(elem, class_type, path="root"):
        if not is_dataclass(class_type):
            return

        expected_fields = set(class_type.__annotations__.keys())
        found_fields = set(child.tag.split('}', 1)[-1] for child in elem if isinstance(child.tag, str))

        for field in expected_fields:
            if field not in found_fields:
                errors.append(f"Missing field '{field}' at {path}")

        for field_name, field_type in class_type.__annotations__.items():
            children = elem.findall(f"DOB:{field_name}", namespaces={"DOB": "http://libcontent.lib.ua.edu/digital/schemas/DOB/dataObject.xsd"})
            for i, child in enumerate(children):
                validate_element(child, field_type, f"{path}/{field_name}[{i}]")

    root_elem = root.find('.//DOB:' + root_class_name, namespaces={"DOB": "http://libcontent.lib.ua.edu/digital/schemas/DOB/dataObject.xsd"})
    validate_element(root_elem, class_defs[root_class_name])

    if errors:
        print("Validation errors found:")
        for err in errors:
            print("  -", err)
    else:
        print("XML matches expected dataclass structure.")

# Print the full class_defs type tree recursively
def print_class_defs_tree(class_defs: Dict[str, Any]):
    def recurse(class_type, indent=0, visited=None):
        if visited is None:
            visited = set()

        typename = class_type.__name__
        if typename in visited:
            print("    " * indent + f"{typename} (recursive reference)")
            return

        visited.add(typename)
        print("    " * indent + typename)
        if not is_dataclass(class_type):
            return

        for field_name, field_type in class_type.__annotations__.items():
            if is_dataclass(field_type):
                print("    " * (indent + 1) + f"{field_name}:")
                recurse(field_type, indent + 2, visited)
            else:
                print("    " * (indent + 1) + f"{field_name}: {field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)}")

    for root_name, root_type in class_defs.items():
        print(f"\nRoot Class: {root_name}")
        recurse(root_type)

# Write nested dataclass objects into the h5 output file
def write_dataclass_to_hdf5(h5group, obj):
    for f in fields(obj):
        val = getattr(obj, f.name)
        if is_dataclass(val):
            subgroup = h5group.create_group(f.name)
            write_dataclass_to_hdf5(subgroup, val)
        elif isinstance(val, list):
            list_group = h5group.create_group(f.name)
            for i, item in enumerate(val):
                if is_dataclass(item):
                    item_group = list_group.create_group(f"{f.name}_{int(i):07}")
                    write_dataclass_to_hdf5(item_group, item)
                else:
                    list_group.attrs[f"{f.name}_{int(i):07}"] = item
        elif val is not None:
            h5group.attrs[f.name] = val

# Load h5 file and print to console
def print_hdf5_tree(hdf5_path: str):
    def recursive_print(name, obj, indent=0):
        prefix = '    ' * indent
        if isinstance(obj, h5py.Group):
            print(f"{prefix}Group: {name}")
            for attr_key, attr_val in obj.attrs.items():
                print(f"{prefix}  [Attr] {attr_key}: {attr_val}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{prefix}Dataset: {name}")
            try:
                value = obj[()]
                print(f"{prefix}  [Data] {value}")
            except Exception as e:
                print(f"{prefix}  [Data] <unreadable: {e}>")
            for attr_key, attr_val in obj.attrs.items():
                print(f"{prefix}  [Attr] {attr_key}: {attr_val}")

    with h5py.File(hdf5_path, 'r') as f:
        f.visititems(lambda name, obj: recursive_print(name, obj, name.count('/')))

# Main function calling block
def main(xsd_path: str, xml_paths: List[str], hdf5_output_path: str, root_class_name: str):
    print("Parsing schema...")
    schema_def = parse_xsd_schema_two_pass(xsd_path)
    print(f"Found schema definitions: {schema_def.keys()}")
    
    print("")
    print("Building dataclasses...")
    class_defs = build_dataclasses_two_pass(schema_def)
    print(class_defs)
    
    #print("")
    #print("validating xml example")
    #validate_xml_structure(ADO_config.xml_test_file_2, class_defs, "dataObject")
    
    print("")
    print_class_defs_tree(class_defs)
    
    print("")
    print("Parsing XML files...")
    objects = []
    for xml_path in xml_paths:
        obj = parse_xml_with_schema(xml_path, class_defs, root_class_name)
        objects.append(obj)
        print(objects)
    
    print("")
    print("Writing HDF5 output...")
    with h5py.File(hdf5_output_path, 'w') as h5f:
        for i, obj in enumerate(objects):
            grp = h5f.create_group(f"object_{int(i):07}")
            write_dataclass_to_hdf5(grp, obj)

    print(f"Done. HDF5 written to {hdf5_output_path}")
    print("")
    
    # print output
    print_hdf5_tree(hdf5_output_path)
    
    timer.stop()
    print(f"Total elapsed time: {int(timer.elapsed())//60}:min {int(timer.elapsed())%60}:sec")
    
main(
    xsd_path= ADO_config.xsd_file_path,
    xml_paths=get_files_by_extension("xml"),
    hdf5_output_path=get_save_path_with_filename(ext="h5", datestamp=True, prompt="Dataset_test_"),
    root_class_name='dataObject'  # must match a complexType name in the .xsd
)

