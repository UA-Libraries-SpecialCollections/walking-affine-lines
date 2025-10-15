#!/usr/bin/python
# OpenAI api uses Python 3.10 :  C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\loc_sort_canit.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# The loc_sort_canit.py script makes a list of dict objects from every line found in all the sort files 
# has the resulting array saved to a .pkl file
# Dict object is keyed as text: "locsh term string", uri: "loc authority uri", embedding: List(float, float, ...) 



from cannery import Cannery  # if saved in a module


cannery = Cannery()
data, path = cannery.pickle(
    all_files=True,
    filetype=".txt",
    harvest_method="line_by_line",
    data_delimited=True,
    delimiter="\t",
    return_type="dict_array"
)
