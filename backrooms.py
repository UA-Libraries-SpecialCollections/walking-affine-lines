#!/usr/bin/python
# import backrooms "S:\Digital Projects\Encoding\testing\backrooms.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# The backrooms.py script provides utility functions for pipeline scripts

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


import tiktoken
import time
import os, codecs, re, unicodedata
from tkinter import Tk, filedialog, simpledialog
from typing import List, Dict, Any, get_origin, get_args
from datetime import datetime
import pickle

class Timer:
    '''
    usage: 
    timer = Timer()
    timer.start()
    do_code.here
    timer.stop()
    print(f"Total elapsed time: {int(timer.elapsed())//60}:min {int(timer.elapsed())%60}:sec")
    '''
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()
        self.end_time = None

    def stop(self):
        self.end_time = time.perf_counter()

    def elapsed(self):
        if self.start_time is None:
            raise ValueError("Timer was never started.")
        end = self.end_time if self.end_time else time.perf_counter()
        return end - self.start_time

def accesspoint(api):
    rekey = ""
    if api == "openai":
        rekey = os.getenv("Key_01")
    return rekey


  
def suffixit(string, suffix):
    if string.endswith(suffix):
        return string[:len(string)-len(suffix)]
    else:
        return string
        
def prefixit(string, prefix):
    if string.startswith(prefix):
        return string[len(prefix):]
    else:
        return string  


def count_tokens(text, model):
    encoding = tiktoken.get_encoding(model) # value for model at this point is 'gpt-3.5-turbo' also the encoding for gpt-4, gpt-3.5-turbo is "cl100k_base"
    tokens_list = encoding.encode(text) # this yields a list object of numeric encodings that reference an identity authority file in chatgpt
    return len(tokens_list)
    
    
def load_text_files_from_directory(ext: str = "txt", mk_pkl: bool = False) -> list[str]:
    """
    Opens a file dialog for the user to select a directory,
    then reads all .txt files in that directory into a list of strings.
    
    optionaly reads in and or makes a python list object pickle file for efficient access to file data
    
    Returns:
       text_data: List[str], A list where each item is the content of one .txt file.
       folder_path: Str, The path to the directory used to collect text from
        
    """
    # Hide the main tkinter window
    root = Tk()
    root.withdraw()
    
    ext = ext.strip('.')
    ext = f".{ext}"
    
    # Ask the user to choose a directory
    folder_path = filedialog.askdirectory(title=f"Select directory containing {ext} files")
    if not folder_path:
        print("No directory selected.")
        return []
    
    text_data = []
    
    # Read all .txt files in the selected directory
    for filename in os.listdir(folder_path):
        if filename.endswith(r".pkl"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "rb") as f:
                    text_data = pickle.load(f)
                    return text_data, folder_path
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    for filename in os.listdir(folder_path):
        if filename.endswith(ext):
            file_path = os.path.join(folder_path, filename)
            try:
                # First try UTF-8
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    text_data.append(text)
            except UnicodeDecodeError:
                try:
                # Fallback to UTF-16
                    with open(file_path, "r", encoding="utf-16") as f:
                        text = f.read()
                        text_data.append(text)
                except Exception as e:
                    print(f"Error reading {filename} with UTF-16: {e}")
            except Exception as e:
                print(f"Error reading {filename} with UTF-8: {e}")
                
    if mk_pkl == True:
        pkl_f_name = f"{os.path.basename(folder_path)}.pkl"
        pkl_path = os.path.join(folder_path, pkl_f_name)
        with open(pkl_path, "wb") as f:
            pickle.dump(text_data, f)
            
    return text_data, folder_path
    
def get_files_by_extension(ext: str) -> List[str]:
    """
    Opens a Tkinter directory selection dialog and returns a list of full file paths
    for all files in the selected directory with the given file extension.

    Parameters:
        ext: File extension to match (e.g., ".txt", ".jpg")

    Returns:
        List of full file paths for files with the given extension.
    """
    # Ensure the extension starts with a dot
    if not ext.startswith("."):
        ext = f".{ext}"

    # Initialize and hide the root Tk window
    root = Tk()
    root.withdraw()

    # Ask the user to select a directory
    directory = filedialog.askdirectory(title=f"Select a folder containing {ext} files")

    if not directory:
        return []

    # Collect all matching files in the directory
    matching_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(ext.lower()) and os.path.isfile(os.path.join(directory, f))
    ]

    return matching_files
    
def get_save_path_with_filename(ext: str = "txt", datestamp: bool = True, prompt: str = "Output_") -> str:
    """
    Opens a Tkinter directory chooser and input prompt to return a full path.

    Parameters:
        ext: String of the desired filename
        datestamp: boolean value for selecting optional disambiguation datestamp string to output filename
        prompt: Text to show in the filename input dialog.

    Returns:
        Full path filename string for 
    """
    # Hide the root Tk window
    root = Tk()
    root.withdraw()

    # Ask the user to select a directory
    directory = filedialog.askdirectory(title="Select a save location")
    if not directory:
        return ""
    
    ext = ext.strip('.')
    ext = f".{ext}"
    
    if datestamp:
        dtn = datetime.now()
        date_val = dtn.strftime("%Y%m%d%H%M%S")
        default_name = f"{prompt}{date_val}{ext}"
    else:
        default_name = f"{prompt}{ext}"
    
    # Ask the user to enter a filename
    filename = simpledialog.askstring(r"Savefile name", r"Use this filename for output file?", initialvalue=default_name, parent=root)
    if not filename:
        return ""

    # Join into a full path
    full_path = os.path.join(directory, filename)
    print(full_path)
    return full_path
    

def load_document_texts_by_prefix(prefix_length: int = 21) -> dict:
    """
    Load and group text files from a directory into a dictionary.
    Files are grouped by a shared filename prefix (first 21 characters by default).

    Parameters:
        prefix_length (int): Number of leading characters used to group files.

    Returns:
        dict: A dictionary with prefix as key and combined file contents as value.
    """
    def _normalize_text(s: str) -> str:
        # drop stray BOMs, normalize unicode + newlines, tame control chars
        s = s.replace('\ufeff', '')
        s = s.replace('\r\n', '\n').replace('\r', '\n')
        s = unicodedata.normalize('NFKC', s)
        # collapse runs of spaces (keep tabs/newlines intact)
        s = re.sub(r'[^\S\n\t]+', ' ', s)
        # remove control chars except \n and \t
        s = re.sub(r'[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]', ' ', s)
        return s.strip()

    def read_ocr_text(file_path: str, normalize: bool = True) -> tuple[str, str]:
        """Return (text, encoding_used) with robust BOM sniffing + sane fallbacks."""
        with open(file_path, 'rb') as fb:
            raw = fb.read()

        # 1) BOM sniff
        if raw.startswith(codecs.BOM_UTF8):
            enc = 'utf-8-sig'
        elif raw.startswith(codecs.BOM_UTF16_LE):
            enc = 'utf-16-le'
        elif raw.startswith(codecs.BOM_UTF16_BE):
            enc = 'utf-16-be'
        elif raw.startswith(codecs.BOM_UTF32_LE):
            enc = 'utf-32-le'
        elif raw.startswith(codecs.BOM_UTF32_BE):
            enc = 'utf-32-be'
        else:
            enc = None

        tried = set()
        if enc:
            try:
                txt = raw.decode(enc)
                return (_normalize_text(txt) if normalize else txt), enc
            except UnicodeError:
                tried.add(enc)

        # 2) Common fallbacks (order matters)
        for enc_try in ('utf-8', 'cp1252', 'latin-1'):
            if enc_try in tried: 
                continue
            try:
                txt = raw.decode(enc_try)
                return (_normalize_text(txt) if normalize else txt), enc_try
            except UnicodeError:
                tried.add(enc_try)

        # 3) Optional: charset-normalizer (if installed)
        try:
            from charset_normalizer import from_bytes
            best = from_bytes(raw).best()
            if best and best.encoding:
                txt = raw.decode(best.encoding, errors='replace')
                return (_normalize_text(txt) if normalize else txt), f"{best.encoding} (detected)"
        except Exception:
            pass

        # 4) Last resort
        txt = raw.decode('utf-8', errors='replace')
        return (_normalize_text(txt) if normalize else txt), 'utf-8 (errors=replace)'
    # Hide the main tkinter window
    
    root = Tk()
    root.withdraw()
    
    
    # Ask the user to choose a directory
    directory_path = filedialog.askdirectory(title=r"(load_document_texts_by_prefix)Select directory containing txt files")
    if not directory_path:
        print("No directory selected.")
        return []
    
    grouped_texts = {}

    for filename in os.listdir(directory_path):
        if not filename.endswith('.txt'):
            continue
        prefix = filename[:prefix_length]
        file_path = os.path.join(directory_path, filename)

        text, used_enc = read_ocr_text(file_path)
        print(f"Reading file: {filename} [encoding={used_enc}]")

        grouped_texts[prefix] = (grouped_texts.get(prefix, "") + " " + text).strip()

    return grouped_texts