#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\sort_aft.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# The sort_aft.py script takes a language extraction file and supports the removal of duplicate line entries
# by sorting all record entries into a text file set using the first two characters of the string as the
# filename.
# Total number of authoritativeLabel elements across all 
# master madsrdf reference files is approximatly 30,000,000. where as a significant portion of those are 
# duplicates.

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


# ------------------------------------------
# Includes

import sys
import tkinter
from tkinter import filedialog
#from filedialog import askopenfilename

# ------------------------------------------
# Get file paths
print("")
print("please select the language_extract_*.txt you want to sort entities from.")
file_path = filedialog.askopenfilename(title='Select language_extract file')
print("")
print(file_path)
print("")

file = file_path.replace("/", "\\")
print(file)


# ------------------------------------------
# Functions

def get_sort_set(line_data):
    entity = line_data.split('\t')[1]
    print(entity)
    if entity[0] in ["\\", r"/", r":", r"*", r"?", "\"", r"<", r">", r"|", r" ", "\n", "\t"]:
        return None
    if entity[1] in ["\\", r"/", r":", r"*", r"?", "\"", r"<", r">", r"|", r" ", "\n", "\t"]:
        return None
    sort_file_path = f"C:\encoding\sort\{entity[:2]}.txt"
    nla = [(line_data.split('\t')[1]).rstrip(), "\t", line_data.split('\t')[0], "\n"] # reorder record values on line to place subject first for faster sorting
    new_line_data = "".join(nla)
    with open(sort_file_path, 'a', encoding='utf-8') as sfile:
        sfile.write(new_line_data)


def sort_file_lines(filepath):
    counter = 0
    with open(filepath, 'r', encoding='utf-8') as dfile:
        for line in dfile: # look at every line in the dupe file
            print(counter)
            counter = counter + 1
            get_sort_set(line)
            

# ------------------------------------------
# Main
sort_file_lines(file)
