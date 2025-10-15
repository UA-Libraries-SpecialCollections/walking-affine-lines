#!/usr/bin/python
# import req_emb "S:\Digital Projects\Encoding\testing\req_emb.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# The req_emb.py script sends embeddings creation requests to the OpenAI API and 
# returns an array of embeddings

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
#import req_emb
#from req_emb import ask_embedding
import numpy as np
from sklearn.cluster import KMeans
import openai
from openai import OpenAI
import tiktoken
from backrooms import accesspoint # comment this line out for git


def ask_embedding(eval_text, s_model, client):
    while True:
        try:
            response = client.embeddings.create(model=s_model, input=eval_text, dimensions=256)
        except Exception as e: # if the openai api returns an error we will wait 1 second and then the try loop will attempt another completion
            print(e)
            time.sleep(1)
            print("Waiting, will retry.")
        else:
            print("embeddings accepted!")
            break # once it succeeds the loop concludes
    embedding = response.data[0].embedding
    return embedding



def ask_prompt(question, temp, prompt_log_file, file_id, testing, s_model, client): # this function calls the ChatGPT API and handles any network "too busy" errors that can make the process fail
    prompt_log_file.write("\n\n" + file_id + "\n" + question + "\n\n") # this logs the item level id and the prompt text to the log file
    if testing == True: # this will exit the function before submitting the chat completion to OpenAI
        return "Test mode equals True"
    while True:
        try:
            response = client.ChatCompletion.create(model=s_model, messages=[{"role": "user", "content": question}], temperature=temp) # here we ask the API for a chat completion object tell it what model we want to use hand our prompt and ocr text over as "question", and set the temp variable
        except Exception as e: # if the openai api returns an error we will wait 1 second and then the try loop will attempt another completion
            print(e)
            time.sleep(1)
            print("Waiting, will retry.")
        else:
            print("Completion accepted!")
            break # once it succeeds the loop concludes
    response_content = response.choices[0].message.content # the returned message object is a dict with the models response in the "content" hash key location. this returned object is a string with the chat response as its value
    response_content = response_content.replace("\n", " ") #these two lines preen the text that is returned from OpenAI to remove any newline characters from the content of the strings. because the presence of those characters within the fields will break the one row per item output metadata
    response_content = response_content.replace("\r", " ")
    return response_content