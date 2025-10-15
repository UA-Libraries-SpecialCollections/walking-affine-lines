#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\assess_txt.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# The assess_txt.py library provides language processing and data composition metrics for a provided
# text. The main function returns a dictionary object keyed by metric name and value. 
#  
# text_assess(): is designed for returning assessment values created from a single body of text

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


# ------------------------------------------
# Includes

import math
import numpy as np
import nltk
from nltk import word_tokenize, sent_tokenize, FreqDist, pos_tag
from typing import Literal
from scipy.special import softmax
from scipy.stats import linregress
import spacy
from spellchecker import SpellChecker
from sklearn.decomposition import PCA
# To install "spellchecker" the following command must be used "pip install pyspellchecker"


# ------------------------------------------
# Download NLTK resources (can be included on the execution scripts first run)                           
nltk.download('punkt') # nltk word and sentence tokenizer models
nltk.download('averaged_perceptron_tagger') # syntactic category tagger for parts-of-speech
nltk.download('stopwords') # nltk corpus file containing common structural words
nltk.download('words') # nltk corpus file. contains ~236,000 english words\
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


# ------------------------------------------
# Embedding analysis metrics to explore
# - For a group of embeddings from a file, item, or aggregate context, measure mean cosine similarity to the centroid. this expresses the relatedness of a cluster of embeddings
# - For a group of segment embeddings arranged across a dimension generate cosine similarity values between each pair in the set and identify the position with the greatest incedent of change. (ex. for a segment across the ten levels of deformation.)
# - Use greatest incedence of change locations to build pairs that hold cross threshold comparitive description
# - For levels of a hierarchy calculate the composite embedding from all embeddings located down the tree. generate and store cosine similarity values between down stream embeddings and composite embeddings. (this will express a segments likeness to the leveled context. (can also be used to identify threshold locations of greatest incedent of change))
# - Use embeddings in ocr transcriptions to generate perplexity comparison values across dimensions i.e. deformation-range-wise, or page-space-wise
# - Use embeddings to compare perplexity between ground truth transcriptions, ocr, and deformation ocr


# ------------------------------------------
# Load spaCy English model (download if not installed)
try:
    nlp = spacy.load("en_core_web_sm")                                                                                  # nlp = a token classifier 
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ------------------------------------------
# Function Definitions


def sum_weighted_max_pooling(
    embeddings: np.ndarray,
    normalization: Literal['max', 'l2', 'softmax', None] = 'max'
) -> np.ndarray:
    """
    Generate a composite embedding by applying sum-weighted max pooling.

    Usage:
        - Max pooling identifies the strongest activation in each embedding dimension.
        - Sum pooling captures the total signal across the cluster (how common or intense
          each feature is across all vectors).
        - Weighting the max-pooled vector with a normalized version of the sum-pooled vector
          emphasizes features that are both strong (max) and consistently present (sum).

    Parameters:
        embeddings (np.ndarray): A 2D array of shape (n_vectors, embedding_dim).
        normalization (str): Method to normalize the sum-pool before weighting the max-pool.
            - 'max':       Scales sum_pool so the largest absolute value is 1. 
                          Use when you want to retain the shape of the signal while bounding it.
            - 'l2':        Scales the sum_pool to unit length using the L2 norm.
                          Use when you care about the direction of the overall cluster signal 
                          and want scale invariance.
            - 'softmax':   Converts the sum_pool into a probability distribution over dimensions.
                          Use when you want to emphasize dimensions with very high frequency
                          while dampening the rest.
            - None:        Use raw (unnormalized) sum values as weights.
                          Use with caution; scale will depend on cluster size and vector magnitudes.

    Returns:
        np.ndarray: A 1D array of shape (embedding_dim,) representing the composite vector.
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array (n_vectors, embedding_dim).")

    sum_pool = np.sum(embeddings, axis=0)
    max_pool = np.max(embeddings, axis=0)

    if normalization == 'max':
        max_val = np.max(np.abs(sum_pool))
        weights = sum_pool / max_val if max_val != 0 else sum_pool

    elif normalization == 'l2':
        norm = np.linalg.norm(sum_pool)
        weights = sum_pool / norm if norm != 0 else sum_pool

    elif normalization == 'softmax':
        weights = softmax(sum_pool)

    elif normalization is None:
        weights = sum_pool

    else:
        raise ValueError("Invalid normalization method. Choose 'max', 'l2', 'softmax', or None.")

    weighted_max = weights * max_pool
    print(f'FINSIHED: WEIGHTED MAX\n')
    return weighted_max



def composite_embedding(embeddings: np.ndarray, pooling: str = 'mean', pca_components: int = 1) -> np.ndarray:
    """
    # function takes a 2D np.ndarray: embeddings array object and a str parameter 'mean', 'max', 'min', 'sum', 'pca'
    # returns a 1D np.ndarray: embedding array object
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if embeddings.ndim != 2:
        raise ValueError("Embeddings array must be 2D (n_samples, embedding_dim).")

    if pooling == 'mean':                                                                                               # creates a new embedding by averaging across each dimension of the embedding stack
        return np.mean(embeddings, axis=0)
        
    elif pooling == 'max':                                                                                              # creates a new embedding by sorting across each dimension of the embedding stack and taking the largest value
        return np.max(embeddings, axis=0)
        
    elif pooling == 'min':                                                                                              # creates a new embedding by sorting across each dimension of the embedding stack and taking the smallest value
        return np.min(embeddings, axis=0)
        
    elif pooling == 'sum':                                                                                              # composite values across each dimension are accumulated into one, preserves the density of dimension within embedding cluster
        return np.sum(embeddings, axis=0)
        
    elif pooling == 'pca':
        if embeddings.shape[0] < pca_components:
            raise ValueError("Number of PCA components cannot exceed number of samples.")
        pca = PCA(n_components=pca_components)
        reduced = pca.fit_transform(embeddings)
        print(f'FINSIHED: COMPOSITE EMBEDDING\n')
        return reduced.flatten()  # Return as 1D vector
        
    else:
        raise ValueError("Invalid pooling method. Use 'mean' or 'max'.")



def text_assess(text):

    # Tokenization
    sentences = sent_tokenize(text)                                                                                     # Sentence Tokenizer, default language = english 
    tokens = word_tokenize(text)                                                                                        # Word Tokenizer, default language = english 
    alpha_tokens = [t.lower() for t in tokens if t.isalpha()]                                                            
    unique_tokens = set(alpha_tokens)                                                                                    

    # POS tagging and lexical features
    pos_tags = pos_tag(alpha_tokens)                                                                                    
    content_words = [word for word, tag in pos_tags if tag.startswith(('NN', 'VB', 'JJ', 'RB'))]                        # Content Words are any words which starts with Noun, Verb (base form), Adjective, or Adverb

    # Metrics
    total_tokens = len(alpha_tokens)                                                                                    
    unique_count = len(unique_tokens)
    try: ttr = unique_count / total_tokens 
    except: 
        ttr = 0 
        print('ttr = Undetermined because of 0 tokens')                                                                                 
    try :avg_token_len = sum(len(w) for w in alpha_tokens) / total_tokens
    except: 
        avg_token_len = 0
        print('avg_token_len = Undetermined because of 0 tokens')                                                       
    avg_sent_len = total_tokens / len(sentences)
    hapax = len([word for word, freq in FreqDist(alpha_tokens).items() if freq == 1])
    hapax_list = [word for word, freq in FreqDist(alpha_tokens).items() if freq == 1]
    try: lexical_density = len(content_words) / total_tokens
    except: 
        lexical_density = 0
        print('lexical_density = Undetermined because of 0 tokens') 
    content_func_ratio = lexical_density / (1 - lexical_density) if lexical_density < 1 else float('inf')               # Percentage of content words/ percentage of non-content words, this helps to model the information density of the text (descriptive language vs simple function language)
       
    # Shannons Entropy
    fdist = FreqDist(alpha_tokens)
    total_freq = sum(fdist.values())
    entropy = -sum((freq / total_freq) * math.log2(freq / total_freq) for freq in fdist.values())                       # Modeling the text's ability to have a word predicted, bits per letter
    
    # Function to compute Zipfian coefficient (slope of log-log rank-frequency)
    def compute_zipf_slope(freq_dist):
        freqs = sorted(freq_dist.values(), reverse=True)
        ranks = np.arange(1, len(freqs) + 1)
        log_ranks = np.log(ranks)
        log_freqs = np.log(freqs)
        try:slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_freqs)                                  # Zipf slope (should be close to -1 for natural languages), close to 0 is noise
        except: 
            slope = -1000
            print('Error with Slope')
        return slope                                                                                                    

    zipf_slope = compute_zipf_slope(fdist)
    
    # Named Entity Recognition function
    def extract_named_entities(text):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]                                                         # Creates a dictionary of text which is detected as an entity 
        return entities

    named_entities = extract_named_entities(text)

    # Word to Word-like ratio function
    def spellcheck_ratio(text):
        spell = SpellChecker()
        tokens = word_tokenize(text.lower())
        wordlike = [t for t in tokens if t.isalpha()]
        if not wordlike:
            return 0.0
        unknown = spell.unknown(wordlike)
        return (len(wordlike) - len(unknown)) / len(wordlike)
        
    splchk_ratio = spellcheck_ratio(text)
    
    # 
    

    # Output
    print(f"Total_Tokens: {total_tokens}")                                  # Total Tokens: Number of all alphabetic tokens in text 
    print(f"Unique_Tokens: {unique_count}")                                 # Unique Count: Number of unique alphabetic tokens in text
    print(f"Type-Token_Ratio: {ttr:.3f}")                                   # TTR (Type Token Ratio): Ratio of unique words by total number of words 
    print(f"Average_Token_Length: {avg_token_len:.2f}")                     # Avgerage Token Length: Average length of word 
    print(f"Number_of_Sentences: {len(sentences)}")                         # Sentences: Number of sentence tokens 
    print(f"Average_Sentence_Length: {avg_sent_len:.2f} tokens")            # Average Sentence Length: Average number of words in sentence
    print(f"Hapax_Legomena_Count: {hapax}")                                 # Hapax: Number of alphabetic tokens that occur only once in text, entity detection 
    print(f"Hapax_Legomena_List: {hapax_list}")                             # Hapax List: Complete list of all alphabetic tokens that occur once, entity detection 
    print(f"Lexical_Density: {lexical_density:.3f}")                        # Lexical Density: Ratio of content words to total number of words, used to categorize by describing the text's feature 
    print(f"Content_Function_Word_Ratio: {content_func_ratio:.3f}")         # Content Function Ratio: Relative amounts of content words to non-content words, used to categorize by describing the text's feature 
    print(f"Shannon_Entropy: {entropy:.3f}")                                # Shannon Entropy: Expression of the predictabiltiy of anyword word to occur in the text, used to characterize noise and structure (natural lanuage model generally trends towards -1)
    print(f"Zipfian_Slope (log-log rank vs freq): {zipf_slope:.3f}")        # Zipfian Slope: Slope of word frequency to the statistical rank of the word (inverse relation of word frequency to word rank), used to characterize noise and structure
    print(f"Word2Wordlike_Ratio: {splchk_ratio:.2f}")                       # Spell Check Ratio: Percent error of words compared to non words, used to determine the quality of the text
    print(f"Total_Entities: {len(named_entities)}")                         # Named Entities: Number of detected  entities 
    print("Named_Entities:")                                                # Returns the dictionary of named entities
    for ent, label in named_entities:
        print(f" - {ent} ({label})")


    assessment_dict = {
    "Total_Tokens": f"{total_tokens}",
    "Unique_Tokens": f"{unique_count}",
    "Type-Token_Ratio": f"{ttr:.3f}",
    "Average_Token_Length": f"{avg_token_len:.2f}",
    "Number_of_Sentences": f"{len(sentences)}",
    "Average_Sentence_Length": f"{avg_sent_len:.2f}",
    "Hapax_Legomena_Count": f"{hapax}",
    "Hapax_Legomena_List": f"{hapax_list}",
    "Lexical_Density": f"{lexical_density:.3f}",
    "Content_Function_Word_Ratio": f"{content_func_ratio:.3f}",
    "Shannon_Entropy": f"{entropy:.3f}",
    "Zipfian_Slope": f"{zipf_slope:.3f}",
    "Word2Wordlike_Ratio": f"{splchk_ratio:.2f}",
    "Total_Entities": f"{len(named_entities)}",
    "Named_Entities": named_entities
    }
    
    return assessment_dict