#!/usr/bin/python

# Take an American Stories dataset pkl file and estimate the cost to generate embeddings and
# assess subject headings using OpenAI completions


import pickle
import backrooms

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import os
from pathlib import Path


# Ask what file to load

load_path = Path(filedialog.askopenfilename(
    title="Load dataset .pkl file...",
    initialdir="C:\\datasets\\american_stories",
    defaultextension=".pkl",
    filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
))


articles = pickle.load(open(load_path, "rb"))



print("")
print("structure of article Dict:")
print("")

for key, value in articles[0].items():
    print(f"{key}: {value}")

print("")
print("Number of articles")
num_arts = len(articles)
print(num_arts)

tokens = 0

for a in articles:
    new_tokens = backrooms.count_tokens(a["article"], "o200k_base") # o200k_base cl100k_base
    tokens = tokens + new_tokens + 500 # +500 is an average number per request covering user prompts, system prompts, and candidates list

print("")
print("Number of tokens")
print(tokens)

t_units = tokens / 1000000

out_tokens_est = num_arts * 183 # estimated average number of tokens per structured response from 30,000 test articles was 183.3333
out_tokens_est = out_tokens_est / 1000000

print("")
print("Number of billable units (per million)")
print(t_units)

print("===============================")
print("===============================")
print("Estimates for GPT5.1")
print("")
i_cost = (250 * t_units) / 100
o_cost = (2000 * out_tokens_est) / 100
t_cost = i_cost + o_cost

print("estimated cost for GPT5.1 ingest tokens")
print(f"${i_cost:.2f}")
print("")
print("estimated cost for GPT5.1 output tokens")
print(f"${o_cost:.2f}")
print("")
print("Total cost for GPT5.1 batch request")
print(f"${t_cost:.2f}")

print("===============================")
print("===============================")
print("Estimates for GPT5 mini")
print("")
i_cost = (45 * t_units) / 100
o_cost = (360 * out_tokens_est) / 100
t_cost_m = i_cost + o_cost

print("estimated cost for GPT5 mini ingest tokens")
print(f"${i_cost:.2f}")
print("")
print("estimated cost for GPT mini output tokens")
print(f"${o_cost:.2f}")
print("")
print("Total cost for GPT5 mini batch request")
print(f"${t_cost_m:.2f}")

print("===============================")
print("===============================")
print("")
print(f"{load_path.name};{num_arts};{tokens};{t_units};{out_tokens_est};${t_cost:.2f};${t_cost_m:.2f}")

# print(articles[0]["article"])