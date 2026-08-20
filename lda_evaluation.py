#!/usr/bin/python
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu



# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


import os
import logging 
import numpy as np
import pandas as pd
import csv
import multiprocessing
from gensim.utils import simple_preprocess
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
from tkinter import filedialog
from scipy.stats import entropy
from scipy.spatial.distance import cosine, jensenshannon
from multiprocessing import freeze_support
from tkinter import filedialog, Tk, simpledialog

logging.basicConfig(level=logging.INFO,format="%(asctime)s : %(levelname)s : %(message)s", datefmt='%m-%d %H:%M', filename=r'S:\Digital Projects\Encoding\testing\lda_testing_evaluationtraining.log', filemode='w')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO) 
formatter = logging.Formatter("%(asctime)s :%(levelname)s : %(message)s")
ch.setFormatter(formatter)
logging.getLogger('').addHandler(ch)
log1 = logging.getLogger('myapp')
logging.basicConfig(filename='training.log',format="%(asctime)s : %(levelname)s : %(message)s", filemode= 'w',  level=logging.INFO)

def kl_divergence(p,q): 
    p = np.array(p) + 1e-12
    q = np.array(q) + 1e-12 
    return np.sum(p* np.log(p/q))

def topic_diversity(lda, topk=10):
    top_words = [word for topic in lda.show_topics(num_topics=-1, num_words=topk, formatted=False) 
                    for word, _ in topic[1]]
    unique_words = len(set(top_words))
    return unique_words / (lda.num_topics * topk)

def doc_score(text, lda, dictionary): 
    tokens = simple_preprocess(text, deacc=True)
    bow = dictionary.doc2bow(tokens)
    topics = lda.get_document_topics(bow, minimum_probability=0.0) 
    dist = np.array([p for _,p in lda.get_document_topics(bow, minimum_probability=0.0)])
    topic_prob = np.zeros(lda.num_topics)
    for topic_id, prob in topics:
        topic_prob[topic_id] = prob
    entro = entropy(dist)
    max_prob = dist.max()
    gini = 1 - np.sum(dist**2)
    num_eff_top = 1 / np.sum(dist**2)
    div = topic_diversity(lda)
    # coh_cv = CoherenceModel(model=lda, texts=[tokens], dictionary=dictionary , coherence='c_v').get_coherence()
    coh_umass = CoherenceModel(model=lda, texts=[tokens], dictionary=dictionary, coherence='u_mass').get_coherence()
    dominance = topic_prob.max()
    kl_div = kl_divergence(topic_prob, np.ones(len(topic_prob))/len(topic_prob))
    perplexity = lda.log_perplexity([bow])
    
    return {
        "dominance": dominance,
        "kl_diver": kl_div,
        "log_perplexity": perplexity,
        "topic_distribution": topic_prob, 
        'coherence_umas': coh_umass, 
        'entropy': entro, 
        'max prob': max_prob, 
        'gini': gini, 
        'num_eff_topics': num_eff_top, 
        'diversity': div, 
        'dist': dist
    }

def main (): 
    root = Tk()
    root.withdraw() 
    directory = filedialog.askdirectory(title='Select Directory for Model, Corpus, Dictionary')
    n_top = simpledialog.askinteger(title= 'Number of Topics', prompt='Enter the number of topics the model was trained on')
    text_path = filedialog.askdirectory(title='Select a directory of text files to run evaluation on corpus')
    root.destroy()
    os.chdir(directory)
    model_dir = os.path.join(directory, f'lda_{n_top}.model')
    dict_dir = os.path.join(directory, 'wiki_dict.dict') 
    csv_dir = os.path.join(directory, 'csv_save')
    save = os.path.join(csv_dir,'model_eval.csv')
    if not os.path.exists(csv_dir): 
        os.mkdir(csv_dir)    
     
    lda = LdaModel.load(model_dir)
    dictionary = Dictionary.load(dict_dir)
    for file in os.scandir(text_path): 
            with open(file.path, 'r', encoding='utf-8') as text: 
                doc_text = text.read() 
                analysis = doc_score(doc_text, lda, dictionary)
                logging.info(analysis)
                results = { 
                        'Text_Name': file.name, 
                        'Dominant Topic Probability': analysis['dominance'],
                        'KL Divergence': analysis['kl_diver'], 
                        'Perplexity': analysis['log_perplexity'], 
                        'Coherence u_mass': analysis['coherence_umas'], 
                        'Entropy': analysis['entropy'], 
                        'Max Topic Prob': analysis['max prob'], 
                        'Gini Index': analysis['gini'], 
                        'Num Effective Topic': analysis['num_eff_topics'], 
                        'Topic Diversity': analysis['diversity']
                    }
                for i, prob in enumerate(analysis['dist']): 
                    results[f'Topic_{i}'] = prob

                df = pd.DataFrame([results])
                if os.path.exists(save): 
                    df.to_csv(save, index=False, mode='a', header=False)
                    logging.info(f'Saved CSV with LDA model analysis of {file.name}')

                else: 
                    df.to_csv(save, index=False, mode='w',header=True)
                    logging.info(f'Saved CSV with LDA model analysis of {file.name}')
 

if __name__ == '__main__':
    freeze_support()
    main() 



