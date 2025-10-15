#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\sh_add_emb.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# The sh_add_emb.py adds text embedding values to the dataset record files
# ingest dataset record files are in the following simple delimited format:
# "subject heading text\tLOC URI\n"
# after adding embedding data:
# "subject heading text\tLOC URI\tembedding data\n"
# export files will 

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


# ------------------------------------------
# Includes

import sys, os, codecs
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


# ------------------------------------------
# make_connection to API
print("setting up API")
connect = "openai" # comment this line out for git
# Set up your OpenAI API credentials
openai.api_key = accesspoint(connect) # in our work flow I import credentials from an external function that stores them in an obfuscated form. # comment this line out for git
# openai.api_key = "REPLACE TEXT BETWEEN QUOTES WITH YOUR API KEY"
selected_model = {"emb_l": "text-embedding-3-large", "emb_s": "text-embedding-3-small", "chat_4": "gpt-4-0125-preview"}


# ------------------------------------------
# Get file paths
print("")
print("please select the dataset text file with LOC headings you want to create embeddings for.")
file_path = filedialog.askopenfilename(title='Select language_extract file')
print("")
print(file_path)
print("")

file = file_path.replace("/", "\\")
print(file)

outfile_path_array = file_path.split(r"/")
filename = outfile_path_array.pop()
filename = filename[:-4] + "_emb" +  ".txt"
outfile_path = "\\".join(outfile_path_array)
outfile = outfile_path + "\\" + filename

print(outfile)

output_obj = open(outfile, 'a', encoding='utf-8')


with open(file, 'r', encoding='utf-8') as hfile:
    for line in hfile:
        h_text = line.split('\t')[0]
        embeddings = ask_embedding(h_text, selected_model["emb_s"])
        output_obj.write(line.rstrip() + "\t" + embeddings + "\n")
        
output_obj.close()