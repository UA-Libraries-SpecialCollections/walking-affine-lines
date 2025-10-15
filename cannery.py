#!/usr/bin/python
# import cannery "S:\Digital Projects\Encoding\testing\cannery.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

# The cannery.py script provides utility functions for making pickle data files from text corpi for various uses

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.


import os
import pickle
import tkinter as tk
from tkinter import filedialog
from typing import List, Optional, Union


class Cannery:
    def __init__(self):
        """Initialize the Cannery and get the directory using Tkinter."""
        root = tk.Tk()
        root.withdraw()
        self.directory = filedialog.askdirectory(title="Select Directory with Text Files")
        if not self.directory:
            raise ValueError("No directory selected.")

    def _make_cache_filename(
        self,
        filetype: str,
        collect_pos: Optional[int],
        delimiter: str
    ) -> str:
        """Generate a unique pickle filename based on config parameters."""
        delimiter_label = {
            "\t": "tab",
            ",": "comma",
            "|": "pipe",
            " ": "space"
        }.get(delimiter, "custom")

        filename = f"harvested_filetype={filetype.strip('.')}_pos={collect_pos}_delim={delimiter_label}.pkl"
        return os.path.join(self.directory, filename)

    def pickle(
        self,
        all_files: bool = True,
        filetype: str = ".txt",
        harvest_method: str = "line_by_line",
        data_delimited: bool = True,
        delimiter: str = "\t",
        collect_pos: Optional[int] = 0,
        return_type: str = "array"
    ) -> Union[List[str], List[List[str]], List[dict]]:
        """
        Harvest and optionally cache processed data from files in a directory.
        If a .pkl file already exists matching the parameters, it loads that.

        Args:
            all_files: If True, process all files in the directory.
            filetype: Extension of files to look for.
            harvest_method: 'line_by_line' currently supported.
            data_delimited: Whether to split each line by a delimiter.
            delimiter: Delimiter character to use for splitting.
            collect_pos: Which position in the split line to collect (used if return_type='array').
            return_type: 'array' (default), or 'dict_array' to return List[Dict] with keys 'text', 'uri', 'embedding'.

        Returns:
            A list of harvested data and the directory path.
        """
        cache_file = self._make_cache_filename(filetype, collect_pos if return_type == "array" else "dict", delimiter)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f), self.directory

        result = []

        for filename in os.listdir(self.directory):
            if not filename.endswith(filetype):
                continue
            file_path = os.path.join(self.directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="utf-16") as f:
                    lines = f.readlines()

            for line in lines:
                if data_delimited:
                    parts = line.strip().split(delimiter)

                    if return_type == "dict_array":
                        if len(parts) >= 3:
                            # Strip brackets and split the embedding
                            embedding_str = parts[2].strip("[]")
                            embedding = [float(x) for x in embedding_str.split(", ") if x]
                            result.append({
                                "text": parts[0],
                                "uri": parts[1],
                                "embedding": embedding
                            })
                    elif collect_pos is not None and len(parts) > collect_pos:
                        result.append(parts[collect_pos])
                else:
                    result.append(line.strip())

        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

        return result, self.directory