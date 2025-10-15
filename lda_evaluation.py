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
    coh_cv = CoherenceModel(model=lda, texts=[tokens], dictionary=dictionary , coherence='c_v').get_coherence()
    coh_umass = CoherenceModel(model=lda, texts=[tokens], dictionary=dictionary, coherence='u_mass').get_coherence()
    dominance = topic_prob.max()
    kl_div = kl_divergence(topic_prob, np.ones(len(topic_prob))/len(topic_prob))
    perplexity = lda.log_perplexity([bow])
    
    return {
        "dominance": dominance,
        "kl_diver": kl_div,
        "log_perplexity": perplexity,
        "topic_distribution": topic_prob, 
        'coherence_cv': coh_cv, 
        'coherence_umas': coh_umass, 
        'entropy': entro, 
        'max prob': max_prob, 
        'gini': gini, 
        'num_eff_topics': num_eff_top, 
        'diversity': div, 
        'dist': dist
    }


def main (text_path): 
    for file in os.scandir(text_path): 
        with open(file.path, 'r', encoding='utf-8') as text: 
            doc_text = text.read() 
            analysis = doc_score(doc_text, lda, dictionary)
            logging.info(analysis)
            results = {
                    "Model": f"{model_type}", 
                    'Text_Name': file.name, 
                    'Dominant Topic Probability': analysis['dominance'],
                    'KL Divergence': analysis['kl_diver'], 
                    'Perplexity': analysis['log_perplexity'], 
                    'Coherence u_mass': analysis['coherence_umas'], 
                    'Coherence c_v': analysis['coherence_cv'], 
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


x = [5,7,10,15]
y = [7]
for i in x: 
    for ii in y: 
        model_title = f'lda_{i}.model'
        model_type = f'{i} topics\\run {ii}'
        logging.info(f'Running {model_title} in {model_type}')
        model_dir = fr"S:\Digital Projects\Encoding\testing\lda_testing_evaluation\{model_type}"
        # model_dir = r'S:\Digital Projects\Encoding\testing\lda_wiki_models'
        # working_dir = r'S:\Digital Projects\Encoding\testing\lda_testing_evaluation'
        lda_dir = os.path.join(model_dir,model_title)
        dic_path = os.path.join(model_dir, "ndnp_dictionary.dict")
        # dic_path = r"S:\Digital Projects\Encoding\testing\lda_wiki_models\wiki_dict.dict"
        text_path = r'S:\Digital Projects\Encoding\testing\lda_testing_evaluation\new_text'
        csv_save = r'S:\Digital Projects\Encoding\testing\lda_testing_evaluation'
        save = os.path.join(csv_save,'NDNP_Model_eval.csv')
        os.chdir(model_dir)



        lda = LdaModel.load(lda_dir)
        dictionary = Dictionary.load(dic_path)

        if __name__ == '__main__':
                    freeze_support()
                    main(text_path=text_path) 



