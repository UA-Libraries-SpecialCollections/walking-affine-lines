#!/usr/bin/python 
# OpenAI api uses Python 3.10 :  C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\assess_txt.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Bryce Shiver 2025 beshiver@crimson.ua.edu

# -------------------------------------Script Summary------------------------------------------
# The lda_model_training.py script is a Latent Dirichlet Allocation (LDA) model training script, it takes a corpus from a wikipedia database in 2017 (4.9M documents) trains an LDA model 
# The script is used to iterate through a set of variables (currently the number of topics in LDA model) and produce LDA models to compare together
# It starts with tokenizing and corpus and dictionary, the variables and input for the tokenizing can be adjusted as well, once tokenized it trains the LDA model on the corpus and dictionary 
# The various LDA models are created and saved 
# After the LDA model creation all models are evaluated and metrics are recored into a text file for each model (metrics are Coherence Score, Topics, Corpus & Dictionary Length) 
# The evaluation creates a visualization for each LDA model as an html 
# Everything is logged and recorded to text file 

# Overall the script is used to train, evaulate, and compare LDA models for topic recognition in a corpus

# -------------------------------------Disclaimer!------------------------------------------
# 
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


# -------------------------------------Includes------------------------------------------
import gensim.downloader as api
import matplotlib.pyplot as plt
import tkinter as tk
import pyLDAvis
import os
import sys
import logging 
import time 
import random
import threading
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary, MmCorpus
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric, remove_stopwords, strip_short
from multiprocessing import freeze_support
from tkinter import filedialog, ttk, messagebox
from pyLDAvis import gensim_models as gensimvis

## -------------------------------Directories----------------------------------------##
# #If Corpus and Dictionary are premade they need to follow the naming convention and location dictated below 
model_dir = filedialog.askdirectory(title='Select Main Directory for Model Saving')
dict_path =os.path.join(model_dir, "wiki_dict.dict")
corpus_path =os.path.join(model_dir, "wiki_corpus.mm")
corpus_save_path = os.path.join(model_dir, "wiki_corpus.mm")
os.chdir(model_dir)

## ---------------------------------Logging--------------------------------------------------## 

def start_logging():
    start = time.time()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s : %(levelname)s : %(message)s", datefmt='%m-%d %H:%M', filename=r'training.log', filemode='w')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s :%(levelname)s : %(message)s")
    ch.setFormatter(formatter)
    logging.getLogger('').addHandler(ch)
    log1 = logging.getLogger('myapp')
    logging.basicConfig(filename='training.log',format="%(asctime)s : %(levelname)s : %(message)s", filemode= 'w',  level=logging.INFO)

## ----------------------------Tokenizing Filters----------------------------------##
# Different filters for tokenizing text in the corpus, include more from gensim.parsing.preprocessing for stricter tokenization
CUSTOM_FILTERS = [
strip_punctuation,
strip_numeric, 
remove_stopwords, 
strip_short
]

## ---------------------------------Training Parameters ---------------------------------##
# Variables for the LDA model to be adjusted, currently being swept through the number of topics 
# Other variables such as Chunk size and Passes can be swept through to compare models 

n_topics = [3,5,10,15,20,30]                                                                                           
chunk_size = 10000                                                                                                                  
total_passes = 1                                                                                                                    
update_every = 1                                                                                                                   

##------------------Text Pre-Processing-----------------------------##
# Creates Tokens from Corpus and 

def preprocess(doc):
    return preprocess_string(doc, CUSTOM_FILTERS)

## -------------------------------Corpus & Dictionary Building-------------------------##
def build_corpus(): 
    logging.info(f"--------------------------------Building Corpus & Dictionary-------------------------------------")
    logging.info("Loading wiki-english-20171001 corpus...")
    wiki_corpus = api.load("wiki-english-20171001")
    dictionary =Dictionary()
    tokenized_texts = []
    start = time.time()

    logging.info("Tokenizing and preprocessing Wikipedia articles...")
    for i, doc in enumerate(wiki_corpus):
            section_text = doc['section_texts']
            compiled_text = " ".join(section_text)
            tokens = preprocess(compiled_text)
        
            if tokens:
                tokenized_texts.append(tokens)

            if i % 100000 == 0 and i > 0:
                logging.info(f"Processed {i:,} articles.")

    dictionary.save(dict_path)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]
    created_corpus = MmCorpus.serialize(corpus_save_path, corpus)    
    end_dict = time.time()

    logging.info(f"[COMPLETE] Saved dictionary with {len(dictionary)} tokens.")   
    logging.info(f'[COMPLETE] Saved Corpus with {len(created_corpus)} ')
    logging.info(f'[TIME] Time for corpus & dictionary: {start-end_dict}')

##-------------------------------LDA Training----------------------------------##

def train_model(item, final_path): 
    freeze_support()  
    logging.info(f"----------------------------[TRAINING] LDA model with {str(item)} Topics-----------------------------------") 
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=item,          
        chunksize=chunk_size,         
        passes=total_passes,          
        random_state=42,
        update_every= update_every
    )
    lda_model.save(final_path)
    logging.info(f'[COMPLETE] {item} Topics Model Saved to: {final_path}')
    return lda_model

## ----------------------------------Evaluation of Model---------------------------------------------------## 

coherence_results=[]

def eval(lda_model, item, corpus_samp): 
    logging.info(f"----------------------------[EVALUATING] LDA model with {str(item)} Topics-----------------------------------") 
    with open (f'lda_model_eval_{str(item)}.txt', 'w') as  f : 

        corpus_len = len(corpus)
        corp_tok = corpus[0][:10]
        dict_len = len(dictionary)

        f.write(f'Corpus Length: {corpus_len}\n')
        f.write(f'Corpus Tokens: {corp_tok}\n')
        f.write(f'Dict Length: {dict_len}\n')

        logging.info(f'Corpus Length: {corpus_len}\n')
        logging.info(f'Corpus Tokens: {corp_tok}\n')
        logging.info(f'Dict Length: {dict_len}\n')

        for idx, topic in lda_model.print_topics(num_topics=item, num_words=30):
            logging.info(f"Topic #{idx}: {topic}\n")
            f.write(f"Topic #{idx}: {topic}\n")
            
        coherence_model = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        coherence_score = coherence_model.get_coherence()
        coherence_results.append((item,coherence_score))
        logging.info(f'Coherence Score: {coherence_score:.4f}\n')
        f.write(f'Coherence Score:{coherence_score}\n')
        
    #     # perplexity = lda_model.log_perplexity(corpus)
    #     # f.write(f'Perplexity Score:{perplexity}\n')
    #     # logging.info(f'Perplexity Score:{perplexity}\n')
    logging.info( 'starting data vis')
    vis_data = gensimvis.prepare(lda_model, corpus_samp, dictionary)
    logging.info('data vis complete')
    pyLDAvis.save_html(vis_data,f'pyLDAvis_html_{str(item)}.html' )

##----------------------------------Plot of Results-----------------------------## 
def eval_plot(): 
    if coherence_results:
        topics, scores = zip(*coherence_results)
        plt.figure(figsize=(10, 6))
        plt.plot(topics, scores, marker='o')
        plt.title('Coherence Score vs Number of Topics')
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence Score (u_mass)')
        plt.grid(True)
        plot_path = os.path.join(model_dir, "coherence_vs_topics.png")
        plt.savefig(plot_path)
        plt.show()
        logging.info(f"[COMPLETE] Saved Coherence Plot to {plot_path}")

##----------------------------Corpus Parsing----------------------------------## 
def sample_corpus_streaming(corpus_path, sample_size=10000, seed=42):
    random.seed(seed)
    reservoir = []
    for i, doc in enumerate(MmCorpus(corpus_path)):
        if i < sample_size:
            reservoir.append(doc)
        else:
            j = random.randint(0, i)
            if j < sample_size:
                reservoir[j] = doc
    return reservoir

##---------------------------Running Functions----------------------------## 
if __name__ == '__main__':  
    if not os.path.exists(model_dir): 
        logging.info('Invalid Model Directory')
        sys.exit(1)

    if not os.path.exists(corpus_path) or not os.path.exists(dict_path): 
        build_corpus()
    start_lda = time.time()

    dictionary = corpora.Dictionary.load(dict_path)
    corpus = MmCorpus(corpus_path)
    sample_corpus = sample_corpus_streaming(corpus_path, sample_size=10000)
    logging.info(f"-------------------------------------Corpus and Dictionary Loaded----------------------------------------\n")
    for item in n_topics: 
        final_path = os.path.join(model_dir, f'lda_{str(item)}.model')
        if not os.path.exists(final_path):
            lda_model = train_model(item=item, final_path= final_path)
            end = time.time()
            logging.info(f'[TIME] Overall Time for LDA Training: {(end-start_lda) // 60,2}')
            eval(lda_model = lda_model, item=item, corpus_samp=sample_corpus)
            logging.info(f'[COMPLETE]Evaluation Complete{str(item)}')

        else:
            start = time.time()
            lda_model = LdaModel.load(final_path)
            end = time.time()
            logging.info(f'[TIME] Overall Time for function: {round((end-start) // 60, 2)}')
            eval(lda_model = lda_model, item=item, corpus_samp=sample_corpus)
            logging.info(f'[COMPLETE] Evaluation Complete{str(item)}')
        
eval_plot()
logging.info(f'[COMPLETE] Training Finished')