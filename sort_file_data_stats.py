#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\sort_file_data_stats.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu


# The sort_file_data_stats.py script generates a report of language statistics from the terms found in a set of sort files.
# script also exports a unique vocabulary list with word frequency as a tab delimited file.

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


import sys, re, os, codecs
from pathlib import Path
import tkinter
from tkinter import filedialog
import tiktoken
import backrooms
from backrooms import count_tokens
#from filedialog import askdirectory

sdir = ""
outputfile = r"C:\encoding\word_histogram.txt"
model = "text-embedding-3-small"

count = 0
t_tokens = 0
vocabulary = set()
histogram = dict()
word_count_per_term = []
token_count_per_term = []

def get_file_dir():
    print("")
    print("please select the directory that contains the alphabetized by first two sort files.")
    sdir_path = filedialog.askdirectory(title='Select Sort File Directory')
    print("")
    print(sdir_path)
    print("")

    sdir = sdir_path.replace("/", "\\")
    print(sdir)
    sdir = Path(sdir)
    return sdir    
    

def get_file_stats(path: Path) -> None:
    global count
    global t_tokens
    global vocabulary
    global histogram
    global word_count_per_term
    global token_count_per_term
    """
    Remove duplicate lines from a single file, preserving order.
    """
    try:
        lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return

    h_text_array = []
    for line in lines:
        count = count + 1
        
        h_text = line.split('\t')[0]
        h_text_array = re.split(r"[ -]", h_text)
        h_text_array = list(filter(None, h_text_array))
        
        t_count = count_tokens(h_text, model)
        token_count_per_term.append(t_count)
        t_tokens = t_tokens + t_count
        
        w_count = len(h_text_array)
        word_count_per_term.append(w_count)
        
        for word in h_text_array:
            word = word.lstrip(r"(")
            word = word.rstrip(r")")
            if word not in vocabulary:
                vocabulary.add(word)
                histogram[word] = 0
            histogram[word] = histogram[word] + 1





dir_path = get_file_dir()


pattern = '*.txt'
txt_files = list(dir_path.glob(pattern))
if not txt_files:
    print(f"No .txt files found in {dir_path}", file=sys.stderr)
    sys.exit(1)

for txt_file in txt_files:
    get_file_stats(txt_file)

with open(outputfile, 'a', encoding='utf-8') as hfile:
    for key in histogram:
        hfile.write(key + "\t" + str(histogram[key]) + "\n")


# ----------------------------------------
print("")
print("total record lines checked")
print(count)
print("")
print("total tokens")
print(t_tokens)
print("")
print("total number of unique words in vocabulary")
print(len(vocabulary))
print("")
print("average number of tokens per term")
print(sum(token_count_per_term)/len(token_count_per_term))
print("")
print("average number of words per term")
print(sum(word_count_per_term)/len(word_count_per_term))
print("")
print("term word count ranged from ")
print(min(word_count_per_term))
print("to")
print(max(word_count_per_term))
print("")
print("word instances in text corpus")
print(sum(word_count_per_term))
print("")
