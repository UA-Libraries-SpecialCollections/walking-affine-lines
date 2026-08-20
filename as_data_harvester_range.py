#!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\as_data_harvester_range.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Jeremiah Colonna-Romano 2025 jjcolonnaromano@ua.edu



"""
Tkinter GUI to download one or more consecutive years from the
'dell-research-harvard/AmericanStories' dataset on Hugging Face
and save them as a single Python pickle file.

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


def download_year_articles(year_str: str):
    """
    Download the tar.gz file for a given year from the Hugging Face dataset
    and return article-level data as a list of dicts.

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
        return None

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
        return None

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

        return articles

    except Exception as e:
        messagebox.showerror(
            "Processing error",
            f"An error occurred while processing the archive for year {year_str}:\n\n{e}"
        )
        return None


def on_download_click(root: tk.Tk, year_var: tk.StringVar, num_years_var: tk.StringVar) -> None:
    """Handle the download button click."""
    year_str = year_var.get().strip()
    num_years_str = num_years_var.get().strip()

    # Validate starting year
    if not year_str:
        messagebox.showerror("Invalid input", "Please enter a starting year.")
        return

    if not year_str.isdigit() or len(year_str) != 4:
        messagebox.showerror(
            "Invalid input",
            "Starting year must be a 4-digit number (e.g., 1809)."
        )
        return

    # Validate number of years
    if not num_years_str:
        messagebox.showerror("Invalid input", "Please enter the number of years to download.")
        return

    if not num_years_str.isdigit():
        messagebox.showerror(
            "Invalid input",
            "Number of years must be a positive integer (e.g., 1, 5, 10)."
        )
        return

    num_years = int(num_years_str)
    if num_years < 1:
        messagebox.showerror(
            "Invalid input",
            "Number of years must be at least 1."
        )
        return

    start_year_int = int(year_str)

    # Build the list of years to download
    years_to_fetch = [str(start_year_int + i) for i in range(num_years)]

    # Check that all requested years are supported by the dataset
    unsupported_years = [y for y in years_to_fetch if y not in SUPPORTED_YEARS]
    if unsupported_years:
        messagebox.showerror(
            "Unsupported year range",
            "The AmericanStories dataset does not have data for these years:\n"
            f"{', '.join(unsupported_years)}\n\n"
            "Supported years are roughly 1770–1774, 1777–1779, 1791–1793, "
            "and 1796–1964 (with some gaps)."
        )
        return

    # Ask where to save the pickle file
    if num_years == 1:
        default_filename = f"AmericanStories_{year_str}.pkl"
    else:
        last_year_int = start_year_int + num_years - 1
        default_filename = f"AmericanStories_{start_year_int}_{last_year_int}.pkl"

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

    # Download and combine all requested years
    root.config(cursor="wait")
    root.update_idletasks()
    try:
        all_articles = []
        for y in years_to_fetch:
            articles = download_year_articles(y)
            if articles is None:
                # Error already shown via messagebox; stop the process
                return
            all_articles.extend(articles)

        # Save combined results
        with open(save_path, "wb") as f_out:
            pickle.dump(all_articles, f_out)

        if num_years == 1:
            msg = (
                f"Downloaded and saved {len(all_articles)} articles for year {year_str}.\n\n"
                f"File saved to:\n{save_path}"
            )
        else:
            last_year_int = start_year_int + num_years - 1
            msg = (
                f"Downloaded and saved {len(all_articles)} articles for years "
                f"{start_year_int}–{last_year_int}.\n\n"
                f"File saved to:\n{save_path}"
            )

        messagebox.showinfo("Success", msg)

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

    # Starting year label + entry
    label_year = ttk.Label(
        main_frame,
        text="Enter a starting year from the AmericanStories dataset (e.g., 1809):"
    )
    label_year.grid(row=0, column=0, sticky="w", pady=(0, 5))

    year_var = tk.StringVar()
    year_entry = ttk.Entry(main_frame, textvariable=year_var, width=10)
    year_entry.grid(row=1, column=0, sticky="w")
    year_entry.focus()

    # Number of years label + entry
    label_num_years = ttk.Label(
        main_frame,
        text="Number of consecutive years to download (e.g., 1, 5, 10):"
    )
    label_num_years.grid(row=2, column=0, sticky="w", pady=(10, 5))

    num_years_var = tk.StringVar(value="1")
    num_years_entry = ttk.Entry(main_frame, textvariable=num_years_var, width=10)
    num_years_entry.grid(row=3, column=0, sticky="w")

    # Button
    download_button = ttk.Button(
        main_frame,
        text="Choose save location and download",
        command=lambda: on_download_click(root, year_var, num_years_var),
    )
    download_button.grid(row=4, column=0, sticky="w", pady=(15, 0))

    root.mainloop()


if __name__ == "__main__":
    build_gui()