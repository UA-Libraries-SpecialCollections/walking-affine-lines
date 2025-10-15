#!/usr/bin/python
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# The dedupe.py script takes a language extraction file and removes any duplicate line entries.
# By building and maintaining a set of alphabetical "seen" list files the large size of a 
# singular reference list is avoided. Total number of authoritativeLabel elements across all 
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

import xml.etree.ElementTree as ET
import sys
import tkinter
from tkinter import filedialog
#from filedialog import askopenfilename

# ------------------------------------------
# Get file paths
print("")
print("please select the language_extract_*.txt you want to remove duplicate entities from.")
file_path = filedialog.askopenfilename(title='Select language_extract file')
print("")
print(file_path)
print("")

file = file_path.replace("/", "\\")
print(file)

outfile_path_array = file_path.split(r"/")
filename = outfile_path_array.pop()
filename = "no_dupe_" + filename[:-4] + ".txt"
outfile_path = "\\".join(outfile_path_array)
outfile = outfile_path + "\\" + filename

sort_base = r"C:\encoding\sort"
sort_file = ""

# ------------------------------------------
# Globals
nd_outfile = open(outfile, 'a', encoding='utf-8')

# ------------------------------------------
# Setup
sort_files_array = [] # this builds the current list of working sort files
with open(r"C:\encoding\sort\current_sort_files.txt", 'r', encoding='utf-8') as scfile:
    for line in scfile:
        sort_files_array.append(line.rstrip())



# ------------------------------------------
# Functions

def get_sort_file(char):
    if char in sort_files_array:
        return f"C:\encoding\sort\{char}.txt"
    if char not in sort_files_array:
        with open(r"C:\encoding\sort\current_sort_files.txt", 'a', encoding='utf-8') as cfile:
            cfile.write(f"{char}\n")
        with open(f"C:\encoding\sort\{char}.txt", 'a', encoding='utf-8') as nfile:
            nfile.close()
        sort_files_array.append(char)
        return f"C:\encoding\sort\{char}.txt"

def get_sort_set(line_data):
    entity = line_data.split('\t')[1]
    print(entity)
    if entity[0] in ["\\", r"/", r":", r"*", r"?", "\"", r"<", r">", r"|", r" ",]:
        return "match"
    sort_file_path = get_sort_file(entity[0])
    with open(sort_file_path, 'r', encoding='utf-8') as sfile:
        for line in sfile:
            if line == line_data:
                return "match"
    with open(sort_file_path, 'a', encoding='utf-8') as afile:
        afile.write(f"{line_data}")
    return "no match"

def dedupe_file_lines(filepath):
    counter = 0
    with open(filepath, 'r', encoding='utf-8') as dfile:
        for line in dfile: # look at every line in the dupe file
            print(counter)
            counter = counter + 1
            if get_sort_set(line) == "no match":
                nd_outfile.write(line)
            


# ------------------------------------------
# Main
dedupe_file_lines(file)
nd_outfile.close()