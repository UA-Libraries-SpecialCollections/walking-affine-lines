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

## ---------------------------------Logging--------------------------------------------------## 
start = time.time()
def setup_logging(model_dir):
    """Set up logging to file in model_dir and to console."""
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(model_dir, "training.log")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove old handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(ch)

    logging.captureWarnings(True)
    logging.info("-------------------------------------------------------------")
    logging.info("LDA Training Session Started")
    logging.info("-------------------------------------------------------------")

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
model_dir = ""
dict_path = ""
corpus_path = ""
corpus_save_path = ""
n_topics = []
chunk_size = 10000
total_passes = 1
iterations = 50
update_every = 1
coherence_results = []
dictionary = None
corpus = None                                                                                                        

##------------------Text Pre-Processing-----------------------------##
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
    MmCorpus.serialize(corpus_save_path, corpus)    
    end_dict = time.time()
    MmCorpus.serialize(corpus_save_path, corpus)
    logging.info(f"[COMPLETE] Saved dictionary with {len(dictionary)} tokens.")   
    logging.info(f'[COMPLETE] Saved Corpus with {len(corpus)} ')
    logging.info(f'[TIME] Time for corpus & dictionary: {start-end_dict}')


# -------------------- Small Sample Corpus Builder -------------------- #
# # Use to Test GUI and overall items 
# def build_corpus():
#     """
#     Build a small sample corpus and dictionary for testing purposes.
#     Saves dictionary and corpus to model_dir.
#     """
#     global dictionary, corpus

#     logging.info("-------- Building Sample Corpus & Dictionary --------")

#     # Sample corpus: list of documents (strings)
#     sample_texts = [
#         "The quick brown fox jumps over the lazy dog",
#         "Machine learning is a fascinating field in computer science",
#         "Artificial intelligence and deep learning are related concepts",
#         "The fox is quick and the dog is lazy",
#         "Data science involves statistics, programming, and domain knowledge",
#         "Python is a popular language for data analysis and machine learning",
#         "Natural language processing allows computers to understand text",
#         "The lazy dog sleeps in the sun",
#         "Deep learning models require large amounts of data",
#         "Libraries like gensim and pyLDAvis help with topic modeling"
#     ]

#     # Preprocess and tokenize
#     tokenized_texts = [preprocess(doc) for doc in sample_texts]

#     # Create dictionary
#     dictionary = Dictionary(tokenized_texts)
#     dictionary.save(dict_path)
#     logging.info(f"[COMPLETE] Sample dictionary saved with {len(dictionary)} tokens at {dict_path}")

#     # Create corpus
#     corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]
#     MmCorpus.serialize(corpus_path, corpus)
#     logging.info(f"[COMPLETE] Sample corpus saved with {len(corpus)} documents at {corpus_path}")

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
        update_every= update_every,
        iterations= iterations, 
        alpha= 'auto', 
        eta= 'auto'
    )
    lda_model.save(final_path)
    logging.info(f'[COMPLETE] {item} Topics Model Saved to: {final_path}')
    return lda_model

## ----------------------------------Evaluation of Model---------------------------------------------------## 
def eval(lda_model, item, corpus_samp): 
    logging.info(f"----------------------------[EVALUATING] LDA model with {str(item)} Topics-----------------------------------") 
    eval_file = os.path.join(model_dir, f'lda_model_eval_{item}.txt')
    with open(eval_file, 'w', encoding='utf-8') as f:   

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

    logging.info( 'starting data vis')
    vis_data = gensimvis.prepare(lda_model, corpus_samp, dictionary)
    logging.info('data vis complete')
    vis_file = os.path.join(model_dir, f'pyLDAvis_html_{item}.html')
    pyLDAvis.save_html(vis_data,vis_file)

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

class LDAGui:
    def __init__(self, root):
        self.root = root
        self.root.title("LDA Trainer GUI")
        self.root.geometry("820x650")

        # Directories
        self.model_dir = tk.StringVar()
        self.n_topics = tk.StringVar()
        self.chunk_size = tk.StringVar()
        self.total_passes = tk.StringVar()
        self.iterations = tk.StringVar()
        self.update_every = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        frame = ttk.LabelFrame(self.root, text="LDA Configuration", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        # Directory selection
        ttk.Label(frame, text="Model Directory:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.model_dir, width=70).grid(row=0, column=1, padx=5, pady=4, sticky="w")
        ttk.Button(frame, text="Browse", command=self.select_model_dir).grid(row=0, column=2, padx=5, pady=4)

        # Add a small separator
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10, pady=(4,8))

        # Parameters block title
        param_block = ttk.LabelFrame(self.root, text="Parameters", padding=8)
        param_block.pack(fill="x", padx=10, pady=(0,10))

        # Define parameters (label text, variable, recommendation)
        params = [
            ("Number of Topics", self.n_topics,
            "Enter single or comma-separated values (e.g. 50,100,200). Recommended: 50–300"),
            ("Chunk Size", self.chunk_size,
            "Recommended: 1,000–10,000"),
            ("Passes", self.total_passes,
            "Recommended: 1–10"),
            ("Iterations", self.iterations,
            "Recommended: 50–100"),
            ("Update Every", self.update_every,
            "Recommended: 1–5")
        ]

        # Build rows: label on left, then an inner frame containing entry + rec label packed tightly
        for row_idx, (label_text, var, rec_text) in enumerate(params, start=0):
            # Left label (param name)
            ttk.Label(param_block, text=label_text + ":").grid(row=row_idx, column=0, sticky="e", padx=(6,8), pady=6)

            # Inner frame to hold entry + recommendation close together
            row_inner = ttk.Frame(param_block)
            row_inner.grid(row=row_idx, column=1, sticky="w", padx=(0,6), pady=6)

            # Entry (packed left)
            entry = ttk.Entry(row_inner, textvariable=var, width=14)
            entry.pack(side="left", padx=(0,4))

            # Recommendation label packed immediately to the right with very small gap
            rec_label = ttk.Label(row_inner, text=rec_text, foreground="gray")
            rec_label.pack(side="left", padx=(2,0))

        # Make the labels column narrow and the inner column expand if window resizes
        param_block.columnconfigure(0, weight=0, minsize=150)
        param_block.columnconfigure(1, weight=1)

        # Progress bars (same as before)
        pb_frame = ttk.LabelFrame(self.root, text="Progress", padding=10)
        pb_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(pb_frame, text="Corpus Creation").pack(anchor="w")
        self.pb_corpus = ttk.Progressbar(pb_frame, length=780, mode="determinate")
        self.pb_corpus.pack(pady=5)
        ttk.Label(pb_frame, text="LDA Training").pack(anchor="w")
        self.pb_lda = ttk.Progressbar(pb_frame, length=780, mode="determinate")
        self.pb_lda.pack(pady=5)

        # Status label
        self.status = tk.StringVar(value="Idle")
        ttk.Label(self.root, textvariable=self.status, foreground="blue").pack(pady=5)

        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10, fill="x", padx=10)
        ttk.Button(btn_frame, text="Build Corpus", command=self.run_build_corpus).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Train LDA", command=self.run_train_lda).pack(side="left", padx=10)
        ttk.Button(btn_frame, text="Quit", command=self.root.quit).pack(side="right", padx=10)
    def select_model_dir(self):
        directory = filedialog.askdirectory(title="Select Main Directory for Model Saving")
        if directory:
            self.model_dir.set(directory)
    
    def run_build_corpus(self):
        threading.Thread(target=self._build_corpus_thread, daemon=True).start()
    
    def run_train_lda(self):
        threading.Thread(target=self._train_lda_thread, daemon=True).start()


    def _build_corpus_thread(self):
        global model_dir, dict_path, corpus_path, corpus_save_path
        self.status.set("Building Corpus...")
        self.pb_corpus["value"] = 0
        self.root.update_idletasks()
        
        model_dir = self.model_dir.get()
        if not model_dir:
            messagebox.showerror("Error", "Please select a model directory.")
            return
        
        setup_logging(model_dir)

        dict_path = os.path.join(model_dir, "wiki_dict.dict")
        corpus_path = os.path.join(model_dir, "wiki_corpus.mm")
        corpus_save_path = corpus_path
        
        build_corpus()
        self.pb_corpus["value"] = 100
        self.status.set("Corpus built successfully!")

    def _train_lda_thread(self):
        global n_topics, chunk_size, total_passes, update_every, dictionary, corpus
        self.status.set("Starting LDA training...")
        self.pb_lda["value"] = 0
        self.root.update_idletasks()

        model_dir = self.model_dir.get()
        if not model_dir:
            messagebox.showerror("Error", "Please select a model directory.")
            return

        setup_logging(model_dir)

        n_topics = [int(t) for t in self.n_topics.get().split(",")]
        chunk_size = int(self.chunk_size.get())
        total_passes = int(self.total_passes.get())
        update_every = int(self.update_every.get())

        dictionary = corpora.Dictionary.load(dict_path)
        corpus = MmCorpus(corpus_path)
        sample_corpus = sample_corpus_streaming(corpus_path, sample_size=10000)

        total = len(n_topics)
        for i, item in enumerate(n_topics):
            final_path = os.path.join(model_dir, f"lda_{item}.model")
            lda_model = train_model(item, final_path)
            eval(lda_model, item, sample_corpus)
            self.pb_lda["value"] = ((i + 1) / total) * 100
            self.root.update_idletasks()

        eval_plot()
        self.status.set("Training and evaluation complete!")
        messagebox.showinfo("Done", "LDA training finished successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = LDAGui(root)
    root.mainloop()