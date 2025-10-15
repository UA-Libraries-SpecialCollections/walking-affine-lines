#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\locsh_collector.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# The locsh_collector.py script parses entity language from authoratativelabel fields and creates a 
# simple flat tab delimited list of LOC entity URI's and entity text

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


import xml.etree.ElementTree as ET
import sys
import tkinter
from tkinter import filedialog
#from filedialog import askopenfilename


print("")
print("please select the *.madsrdf.xml you want to extract entities from.")
file_path = filedialog.askopenfilename(title='Select madsrdf file')
print("")
print(file_path)
print("")

file = file_path.replace("/", "\\")
print(file)

outfile_path_array = file_path.split(r"/")
filename = outfile_path_array.pop()
filename = "language_extract_" + filename[:-4] + ".txt"
outfile_path = "\\".join(outfile_path_array)
outfile = outfile_path + "\\" + filename

print(outfile)

# Parse XML file
tree = ET.parse(file)
root = tree.getroot()

# Namespaces used in the XML (adjust as needed)
namespaces = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'madsrdf': 'http://www.loc.gov/mads/rdf/v1#'
}

# Array to store results
collected_data = []

# Iterate through all rdf:Description elements
for desc in root.findall('.//rdf:Description', namespaces):
    rdf_about = desc.get(f"{{{namespaces['rdf']}}}about")
    auth_label_elem = desc.find('madsrdf:authoritativeLabel', namespaces)
    
    if rdf_about is not None and auth_label_elem is not None:
        print("!!! in append loop!!!")
        collected_data.append([rdf_about, auth_label_elem.text])

# Write collected data to output file
with open(outfile, 'w', encoding='utf-8') as f:
    for item in collected_data:
        f.write(f"{item[0]}\t{item[1]}\n")