#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\add_emb_to_sort_files.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu


# The add_emb_to_sort_files.py script opens and adds embeddings to lines found in any .txt files found in the selected directory

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


import sys, os, codecs
from pathlib import Path
import tkinter
from tkinter import filedialog
import time
import req_emb
from req_emb import ask_embedding
import numpy as np
from sklearn.cluster import KMeans
import openai
import tiktoken
from backrooms import accesspoint # comment this line out for git
from BERT_xml_skeleton_populator import _embed_one



# ------------------------------------------
# make_connection to API
print("setting up API")
connect = "openai" # comment this line out for git
# Set up your OpenAI API credentials
openai.api_key = accesspoint(connect) # in our work flow I import credentials from an external function that stores them in an obfuscated form. # comment this line out for git
# openai.api_key = "REPLACE TEXT BETWEEN QUOTES WITH YOUR API KEY"
selected_model = {"emb_l": "text-embedding-3-large", "emb_s": "text-embedding-3-small", "chat_4": "gpt-4-0125-preview"}


sdir = ""
count = 0

def make_embedding(transcript):

    vec = _embed_one(transcript, mode="passage")
    return vec

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

def process_file(path: Path) -> None:
    global count
    """
    add embeddings to lines from a single file, preserving order.
    """
    try:
        lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return


    embedded = []
    for line in lines:
        count = count + 1
        h_text = line.split('\t')[0]
        embeddings = make_embedding(h_text)
        print("")
        print(h_text)
        print(embeddings)
        print("")
        emb_line = line.rstrip() + "\t" + str(embeddings) + "\n"
        embedded.append(emb_line)


    try:
        path.write_text(''.join(embedded), encoding='utf-8')
    except Exception as e:
        print(f"Error writing {path}: {e}", file=sys.stderr)



dir_path = get_file_dir()
pattern = '*.txt'
txt_files = list(dir_path.glob(pattern))
if not txt_files:
    print(f"No .txt files found in {dir_path}", file=sys.stderr)
    sys.exit(1)

 
for txt_file in txt_files:
    process_file(txt_file)


print("total record lines updated")
print(count)
