#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\as_data_harvester.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu

"""
Tkinter GUI to download a single year from the
'dell-research-harvard/AmericanStories' dataset on Hugging Face
and save it as a Python pickle file.

It *does not* use `datasets.load_dataset` and therefore avoids
the `trust_remote_code` issue in datasets>=4.0.0.

Requirements:
    pip install huggingface_hub
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import json
import tarfile
import pickle

# Years supported by the AmericanStories dataset (from AmericanStories.py on HF)
SUPPORTED_YEARS = [
    "1770", "1771", "1772", "1773", "1774",
    "1777", "1778", "1779",
    "1791", "1792", "1793",
] + [str(year) for year in range(1796, 1964 + 1)]  # 1796–1964 inclusive


def download_and_save_year(year_str: str, save_path: str) -> None:
    """
    Download the tar.gz file for a given year from the Hugging Face dataset
    and save article-level data as a pickle (list of dicts).

    Each dict roughly matches the HF loader's 'associated' article format:
    {
        "article_id", "newspaper_name", "edition",
        "date", "page", "headline", "byline", "article"
    }
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as e:
        messagebox.showerror(
            "Missing dependency",
            "You must install the 'huggingface_hub' package first:\n\n"
            "    pip install huggingface_hub\n\n"
            f"Details: {e}"
        )
        return

    # 1) Download the year's tarball from the dataset repo
    try:
        tar_path = hf_hub_download(
            repo_id="dell-research-harvard/AmericanStories",
            filename=f"faro_{year_str}.tar.gz",
            repo_type="dataset",
        )
    except Exception as e:
        messagebox.showerror(
            "Download error",
            f"Failed to download data for year {year_str} from Hugging Face:\n\n{e}"
        )
        return

    # 2) Parse the tarball and extract article-level data
    articles = []
    try:
        year_dir_prefix = f"faro_{year_str}"
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                filepath = member.name  # e.g. "faro_1809/18300101_0001_ed-01_...json"
                if not filepath.startswith(year_dir_prefix):
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue

                try:
                    data_bytes = f.read()
                    data = json.loads(data_bytes.decode("utf-8"))
                except Exception:
                    # Skip files that fail to decode/parse
                    continue

                # This matches the "associated=True" branch from AmericanStories.py
                if "lccn" not in data or "full articles" not in data:
                    continue

                filename = filepath.split("/")[-1]        # drop directory prefix
                name_no_ext = filename.split(".")[0]      # drop ".json"
                parts = name_no_ext.split("_")

                # Expect something like: DATE_PAGE_<...>_editionXXXX_...
                if len(parts) < 3:
                    continue

                scan_date = parts[0]          # e.g. "18300101"
                scan_page = parts[1]          # e.g. "0001"
                scan_edition_raw = parts[-2]  # e.g. "edition01" (from loader script)

                # The original loader uses scan_edition = filepath.split("_")[-2][8:]
                # to strip "edition" from "edition01".
                scan_edition = (
                    scan_edition_raw[8:] if len(scan_edition_raw) > 8 else scan_edition_raw
                )

                scan_id = name_no_ext
                newspaper_name = data.get("lccn", {}).get("title", "")

                for article in data.get("full articles", []):
                    full_article_id = article.get("full_article_id")
                    article_id = f"{full_article_id}_{scan_id}"

                    articles.append(
                        {
                            "article_id": article_id,
                            "newspaper_name": newspaper_name,
                            "edition": scan_edition,
                            "date": scan_date,
                            "page": scan_page,
                            "headline": article.get("headline"),
                            "byline": article.get("byline"),
                            "article": article.get("article"),
                        }
                    )

        # 3) Save as a pickle file
        with open(save_path, "wb") as f_out:
            pickle.dump(articles, f_out)

        messagebox.showinfo(
            "Success",
            f"Downloaded and saved {len(articles)} articles for year {year_str}.\n\n"
            f"File saved to:\n{save_path}"
        )

    except Exception as e:
        messagebox.showerror(
            "Processing error",
            "An error occurred while processing the archive or saving the pickle file:\n\n"
            f"{e}"
        )


def on_download_click(root: tk.Tk, year_var: tk.StringVar) -> None:
    """Handle the download button click."""
    year_str = year_var.get().strip()

    if not year_str:
        messagebox.showerror("Invalid input", "Please enter a year.")
        return

    if not year_str.isdigit() or len(year_str) != 4:
        messagebox.showerror(
            "Invalid input",
            "Year must be a 4-digit number (e.g., 1809)."
        )
        return

    if year_str not in SUPPORTED_YEARS:
        messagebox.showerror(
            "Unsupported year",
            "The AmericanStories dataset does not have data for this year.\n\n"
            "Supported years are roughly 1770–1774, 1777–1779, 1791–1793, "
            "and 1796–1964 (with some gaps)."
        )
        return

    # Ask where to save the pickle file
    default_filename = f"AmericanStories_{year_str}.pkl"
    save_path = filedialog.asksaveasfilename(
        parent=root,
        title="Save dataset as...",
        defaultextension=".pkl",
        initialfile=default_filename,
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
    )

    if not save_path:
        # User cancelled
        return

    # This may take a while depending on the year and connection
    root.config(cursor="wait")
    root.update_idletasks()
    try:
        download_and_save_year(year_str, save_path)
    finally:
        root.config(cursor="")
        root.update_idletasks()


def build_gui() -> None:
    root = tk.Tk()
    root.title("AmericanStories Year Downloader")

    # Main frame
    main_frame = ttk.Frame(root, padding=15)
    main_frame.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Label
    label = ttk.Label(
        main_frame,
        text="Enter a year from the AmericanStories dataset (e.g., 1809):"
    )
    label.grid(row=0, column=0, sticky="w", pady=(0, 5))

    # Entry
    year_var = tk.StringVar()
    year_entry = ttk.Entry(main_frame, textvariable=year_var, width=10)
    year_entry.grid(row=1, column=0, sticky="w")
    year_entry.focus()

    # Button
    download_button = ttk.Button(
        main_frame,
        text="Choose save location and download",
        command=lambda: on_download_click(root, year_var),
    )
    download_button.grid(row=2, column=0, sticky="w", pady=(10, 0))

    root.mainloop()


if __name__ == "__main__":
    build_gui()