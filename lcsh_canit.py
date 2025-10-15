#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\lcsh_canit.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# The lcsh_canit.py script makes a flat list of all the terms from the sort files 
# has the resulting array saved to a .pkl file

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.



from cannery import Cannery  # if saved in a module
#from topic_modeling import train_and_save_lda_from_lcsh, preprocess_lcsh
from datetime import datetime
import os

cannery = Cannery()
data, path = cannery.pickle(
    all_files=True,
    filetype=".txt",
    harvest_method="line_by_line",
    data_delimited=True,
    delimiter="\t",
    collect_pos=0,
    return_type="array"
)
'''
dtn = datetime.now()
moment = dtn.strftime("%Y%m%d%H%M%S")
mf = f"LDA_model_{moment}.model"
df = f"LDA_dict_{moment}.dict"


train_and_save_lda_from_lcsh(
    lines=data,
    num_topics=75,
    model_save_path=os.path.join(path, mf),
    dictionary_save_path=os.path.join(path, df)
)
'''