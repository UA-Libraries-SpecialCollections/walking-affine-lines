#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\assess_txt.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Bryce Shiver 2025 beshiver@crimson.ua.edu

# The cultivate_dataset.py script allows for the extraction of text and embeddings from an h5 file 
# It utilizes the assess_txt library to help process the transcritions and analysis text and provide data about the text 
# This script reads the transcriptions from hdf5 file and processes them, and finally saves the output of the analysis back to the hdf5 file 
# It can iterate through a directory of hdf5 files 
# The script will also create a map of lowest level attributes (with path) of the hdf5 file to aid with traversing through hdf5 file 
# This script needs text in an hdf5 file to work properly 


# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


# ------------------------------------------
# Includes

import assess_txt 
import nltk
import os 
import h5py 
import tkinter as tk 
import numpy as np 
from hdf5reader import hdf5_text
import spacy

# ------------------------------------------
# Downloads needed to process the text corpus in the assess_txt library 
nltk.download('punkt') # nltk word and sentence tokenizer models
nltk.download('averaged_perceptron_tagger') # syntactic category tagger for parts-of-speech
nltk.download('stopwords') # nltk corpus file containing common structural words
nltk.download('words') # nltk corpus file. contains ~236,000 english words\
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
# ------------------------------------------

hdf5_dir = tk.filedialog.askdirectory(title = "Select Directory of HDF5")                           # Requests directory of the hdf5 files 
os.chdir(hdf5_dir)                                                                                  # Changes directory to hdf5 directory 

# nest = []                                                                                         # Creates empty list to store nested embedding arrays                                                                              
found_attr = []                                                                                     # Creates empty list to store found names of attributes 
map =[]                                                                                             # Creates empty list to store the final path and name of attributes 

def text_data(name, obj):                                                                           # Function used to process hdf5 files, sort through groups, datasets, and attributes to find the correct source and perform text analysis and array transforms, then saves back to hdf5 file 
    


 # Prints out Groups, Datasets, and Attributes as well as creating a map of hdf5 file with a layout of the lowest level attributes 
    if isinstance(obj, h5py.Group):                                                                 # Prints out all groups in hdf5 file and writes all attributes to the keys variable 
        # print(f"\n Group: {name}")                                                                # Uncomment to print all group values 
        keys = list(obj.attrs.keys())

    elif isinstance(obj, h5py.Dataset):                                                             # Prints out all datasets in the file 
        print(f"\n Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")   

    if keys:                                                                                        # Searches for any keys in the keys list 
            # print(f'\n Attributes: {name}\{keys}')                                                # Uncomment to print all attribute names 
            
            for attr_name in keys:                                                                  # For any keys it will search for the attribute name 
                if attr_name not in found_attr:                                                     # If a unique attribute is not found it is printed and appenended to the list 
                    full_name = str(name) + f'/'+ str(attr_name)                                    # Creates string of the path plus the attribute name for human readability 
                    found_attr.append(attr_name)                                                    # Appends list of previously used attributes so they are only recorded once for the map 
                    map.append(full_name)                                                           # Appends the list to print to text file 
        

  # Creates a group called analytics in the hdf5 file to store the text analysis results
 
    if isinstance(obj, h5py.Group) and 'segmentVectorEmbedding' in obj:                             # Searches groups to find one named 'segmentVectorEmbedding'
        try: 
            obj.create_group('analytics')                                                           # Creates 'analytics' group in the sibling level of 'segmentVectorEmbedding' 
            print(f'Group created under: {name}\n')

        except: 
            print(f'Group already made: {name}')


    
  # Extracts and processes text in 'transcription' attributes with the assess.txt library 
        try: 
            if isinstance (obj, h5py.Group ) and 'transcription' in obj.attrs :                     # Searches group and attributes to find ones named 'transcription'
                    print(f'\n --- {name} being processed ---\n')
                    transcript_data = obj.attrs['transcription']                                    # Extracts text from the found attribute and then allows it to be processed                                          
                    print(f'\nTranscription:\n{transcript_data}\n')                                      
                    results = assess_txt.text_assess(transcript_data)                               # Uses the assess_text to get text corpus metrics on the text in the attribute 
                    print(f'\n ---- Text Assessed: {name} ----\n')

        except Exception as e: 
             print(f'Failure in transcription analysis for :{name}\n ERROR: {e} ')


 # Writes results from assess_txt library to the 'analytics' group, each value in the assess_txt dictionary is written as a different attribute 
        try: 
            if isinstance(obj, h5py.Group) and 'analytics' in obj:                                  # Searches through groups to find the 'analytics' attribute in which to write the results 
                analytics = obj['analytics']
                for k, v in results.items():                                                        # Uses every key (k) and value (v) in the dictionary from the assess_txt dictionary to write to the 'analytics' attribute 
                    analytics.attrs[k] = v                                                          # Writes the analytics attribute with the name as key and the value under the name 
                print(f'\nAttribute Written in: {name}\n')

        except Exception as e: 
             print(f'Failure in writing results to analytics :{name}\n ERROR: {e}')



 # Extracts embeddings from h5 file, converts them from a string to an array then creates a dataset ('s_embed_array') to save the embedding array to the h5 file 
    # try: 
    #     if isinstance(obj, h5py.Group) and 'segmentVectorEmbedding' in obj:                         # Filters through groups to find the 'segmentVectorEmbedding' to extract the embedding 
    #                 seg_vec = obj['segmentVectorEmbedding']                                         # Sets a variable to the group which was filtered
    #                 print(f"found 'segmentVectorEmbedding' in: {obj.name}")


    #                 if 's_embedding' in seg_vec.attrs:                                              # Filters through the group attributes for the 's_embedding' attribute which contains the values of the embedding string 
    #                     print(f"found 's_embedding' in: {obj.name}") 

    #                     try: 
    #                         embedding = seg_vec.attrs['s_embedding']                                # Sets a variable to the found attribute above
    #                         values = [embedding.split(';')]                                         # Splits the embedding string (str) to a list 
    #                         emb_arr = np.array(values,dtype=np.float32)                             # Turns the list to a numpy array to post it to the h5 file 

    #                         if 's_embed_array' in seg_vec:                                          # Searches for 's_embed_array' in the group 
    #                             del seg_vec['s_embed_array']                                        # If the name is in the group it deletes it, to ensure the dataset can be created 

    #                         seg_vec.create_dataset('s_embed_array', data=emb_arr.T)                 # Creates a dataset in the group found at the top, named: 's_embed_array', the data is from the embedding array of all of the embedding values 
    #                         print(f'saved sembedding to: {seg_vec.name}')

    #                     except Exception as e: 
    #                         print(e)
    #                 else: 
    #                     print(f"'s_embedding' NOT found in: {obj.name}") 
        
    #                 nest.append(values)                                                             # Adds the value of the embedding list to the overall list of embedding values 

    # except Exception as e: 
    #         print(f'Error in text_data processing: {e}')


              
completed = []                                                                                      # Creates a empty list that can be used for a list of completed files 
ext = ['.h5', '.hdf5' ]                                                                             # List of possible endings of hdf5 file to reduce error of reading files 

# Sorts through all hdf5 files in a directory to open and then use the function (text_data) on every file and 
for hdf5 in os.scandir(hdf5_dir):                                                                   # Walks through all files in a specified directory 
    if hdf5.is_file() and hdf5.name.endswith(tuple(ext)):                                           # Ensures file in directory is an hdf5 file 
        print(f'\n--- {hdf5.path} ---')                                                             # Prints path 
        with h5py.File(hdf5, 'r+') as f:                                                            # Opens the hdf5 file with the h5py library with read and write access 
            f.visititems(text_data)                                                                 # Uses the visititems function in the h5py lib, in the text_data function, it requires a callable function to work 
    
            
            print(f'\n--- {hdf5.name} has been processed ---\n')
            completed.append(hdf5.name)                                                             # Adds file name to the list of completed files 

# Creates an text file to record map of hdf5 files 
with open ('hdf5_map.txt', 'w') as m:                                                               # Opens file                                                 
    m.write(f' Overall Map of HDF \n \n --list contains each unique lowest level of hdf5 file-- \n ')   # Writes string to file 
    for items in map:                                                                               # Iteratior to move through all items in list and print them to text file 
        m.write(f'\n {items}')                                                                      # Writes every item from list to new line of the file
    print(f'-- File Written Successfully -- \n')                                                    # Print out stating the process is finished 
m.close()                                                                                           # Closes texzt file 
                                                                    
# # Final 2D numpy array 

# try: 
#             embed_array = np.array(nest, dtype=np.float32)
#             print(embed_array)
#             shape = np.shape(embed_array)
#             print(shape)

# except Exception as e: 
#             print(f'error converting to Numpy array: {e}')


# # Running the embedding analysis on the 
# assess_txt.sum_weighted_max_pooling(embed_array)
# assess_txt.composite_embedding(embed_array)


print(f'Processed Files: {completed} \n')                                                           # Prints completed list of all files processes 

             

    

