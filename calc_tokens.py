#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\calc_tokens.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# The calc_tokens.py script takes a language extraction file and sums the total number
# of ingest tokens its text corpus contains in preperation for LLM processing

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


import tiktoken

file_path = r"C:\encoding\locsh_language_extract.txt"

#embedding models pricing as of may-7-2025, text-embedding-3-small: $0.02, text-embedding-3-large: $0.13, per million tokens
model = "text-embedding-3-small" #"cl100k_base"
tokens = 0

def count_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model) # value for model at this point is 'gpt-3.5-turbo' also the encoding for gpt-4, gpt-3.5-turbo is "cl100k_base"
    tokens_list = encoding.encode(text) # this yields a list object of numeric encodings that reference an identity authority file in chatgpt
    return len(tokens_list)


with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()


    for line in lines:
        record = line.split('\t')
        string = record[1]
        string = string.rstrip()
        tokens = tokens + count_tokens(string, model)

print("number of tokens = ")
print(tokens)