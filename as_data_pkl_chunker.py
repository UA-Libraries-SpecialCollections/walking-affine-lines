#!/usr/bin/python

# take american stories dataset pkl file and make a testing set of the first n records

import pickle
import backrooms

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import os
from pathlib import Path


# Ask what file to load

load_path = Path(filedialog.askopenfilename(
    title="Load large dataset .pkl file...",
    initialdir="C:\\datasets\\american_stories",
    defaultextension=".pkl",
    filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
))

num_set = 100

save_path = Path(f"{load_path.parent}/{load_path.stem}_{num_set}{load_path.suffix}")
print(save_path)


articles = pickle.load(open(load_path, "rb"))


# Save combined results
with open(save_path, "wb") as f_out:
    pickle.dump(articles[:num_set], f_out)

