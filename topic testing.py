#!/usr/bin/python 
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\assess_txt.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Bryce Shiver 2025 beshiver@crimson.ua.edu
# -------------------------------------Code Summary------------------------------------------



# -------------------------------------Disclaimer!------------------------------------------
# 
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


# -------------------------------------Includes------------------------------------------
from gensim import corpora
from gensim.corpora import Dictionary, MmCorpus
import tkinter as tk
from tkinter import filedialog
from gensim.models import LdaModel
import os
import logging 
import time 
import re
from openai import OpenAI
import csv
import backrooms


## ---------------------------------Directories & Inputs-----------------------------------------## 
model_dir = filedialog.askdirectory(title='Select Main Directory for Model Saving')
os.chdir(model_dir)
csv_path = os.path.join(model_dir, 'csv')
summ_path = os.path.join(model_dir,'summaries')
comp_path = os.path.join(model_dir, 'list')
csv_files = [f for f in os.scandir(csv_path) if f.is_file() and f.name.endswith('.csv')]



n_topics = 5

model_name = f'lda_{n_topics}'
weight_threshold = [0.0005]

## -----------------------------------OpenAI Setup-------------------------------------## 
key = backrooms.accesspoint('openai')
client = OpenAI(api_key = key)
gpt_model= 'gpt-5-nano'
request = """
You are given a list of words representing the top terms in an LDA topics trained from a set of newspapers.
Summarize the list into a short, clear topic heading (1-4 words or a short phrase) that best describes the overall theme.
The list of words is descending in importance for the topic. 
When analyzing the topics think of titles related to topics in a newspaper.
Return only the heading, nothing else.
Example:
Topic words: Japan, Japanese, Tokyo, Prefecture, anime, Osaka, manga, TBS, NHK, Fuji, theme, Prefectural, Asahi, NTV, Nippon
Heading: Japanese Culture & Media
Now summarize this list:
Topic words: """


##------------------------------------------Logging--------------------------------------------## 
start = time.time()
logging.basicConfig(level=logging.INFO,format="%(asctime)s : %(levelname)s : %(message)s", datefmt='%m-%d %H:%M', filename=r'training.log', filemode='w')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO) 
formatter = logging.Formatter("%(asctime)s :%(levelname)s : %(message)s")
ch.setFormatter(formatter)
logging.getLogger('').addHandler(ch)
log1 = logging.getLogger('myapp')
logging.basicConfig(filename='training.log',format="%(asctime)s : %(levelname)s : %(message)s", filemode= 'w',  level=logging.INFO)

##-----------------------------------------Functions-----------------------------------------## 
def sort_key(f): 
    m = re.search(r'\.topic(\d+)\.(\d*\.?\d+)weights\.csv$', f.name)
    if m:
        topic_id = int(m.group(1))
        weight = float(m.group(2))
        return (topic_id, weight)
    return (0, 0)

def topic_words(lda_model, topic_id, max_weight, n_topics ): 
    word_probs = lda_model.get_topic_terms(topicid=topic_id, topn=n_topics)
    words_weights = [(lda_model.id2word[word_id], weight) for word_id, weight in word_probs]
    word_list =[]
    for word, weight in words_weights:                    
        if weight > max_weight: 
            word_list.append((word, weight))
    return(word_list)

def make_csv(lda_model, num_top, weight_threshold, model_name, csv_path):
    logging.info(f"----------------------------Making CSVs-----------------------------------") 
    for topic_id in range(num_top):
        word_list = topic_words(lda_model=lda_model, topic_id=topic_id, max_weight=weight_threshold, n_topics=num_top)
        made_csv = os.path.join(csv_path,f'{model_name}.topic0{int(topic_id)}.{weight_threshold}weights.csv' )
        with open (made_csv, 'w', encoding='utf-8', newline='') as  f :

            logging.info(f'Making CSV file')
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            for word, weight in word_list: 
                wr.writerow([word,weight])
            logging.info(word_list)
    sorted_file = sorted(csv_path)

def CHATGPT(update_prompt, gpt_model, client, weight_threshold, topic_id, summ_path, comp_path): 
    logging.info(f'Running the Prompt: {update_prompt}')
    logging.info('OPENING THE ROBOT')
    response = client.chat.completions.create(
    model=gpt_model, 
    messages=[ 
        {'role':'user', 'content': f'{update_prompt}'}
        ] 
    )
    output = response.choices[0].message.content
    logging.info(f'Resulting Output = {output}')
    summary_save = os.path.join(summ_path,f'{model_name}_topic{int(topic_id)}.txt' )
    with open (summary_save, 'w') as  summary:
        summary.write(output)
        logging.info(f'Response written to {summary}')

    comp_save = os.path.join(comp_path, f'{model_name}_topic_list.txt' )
    with open (comp_save, 'a') as file:
        file.write(f'Topic {topic_id}:  {output}\n')


def summary(request,gpt_model, client, csv_path, summ_path, comp_path):
    logging.info(f"----------------------------Summarizing Topics-----------------------------------")
    topic_response ={}
    for file in  os.scandir(csv_path):

        if file.is_file() and file.name.endswith('csv'):
            m = re.search(r'\.topic(\d+)\.(\d*\.?\d+)weights\.csv$', file.name)
            if not m: 
                continue      
            topic_id = int(m.group(1))
            weight_threshold = float(m.group(2)) 

            with open (file.path, encoding='utf-8', newline='') as  f :
                logging.info(f'opening {f}....')
                text = csv.reader(f, quoting=csv.QUOTE_ALL)
                word_list = [row[0] for row in text if row]

            update_prompt = (f'{str(request)} {word_list}')
            CHATGPT(update_prompt=update_prompt, gpt_model=gpt_model, client=client, weight_threshold=weight_threshold,  topic_id=topic_id, summ_path=summ_path, comp_path =comp_path)
                    
##--------------------------------------Working---------------------------------------------## 
final_path = os.path.join(model_dir, f'{model_name}.model')
lda_model = LdaModel.load(final_path)

for items in weight_threshold: 
        make_csv(lda_model= lda_model, num_top=n_topics, weight_threshold=items, model_name=model_name, csv_path = csv_path)

sorted_files = sorted(csv_files, key=sort_key)
summary(request=request, gpt_model=gpt_model, client=client, csv_path = csv_path, summ_path=summ_path, comp_path = comp_path)