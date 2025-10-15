import os 
import tkinter 
from tkinter import filedialog
from pathlib import Path
from glob import glob
import logging
from pypdf import PdfReader
from collections import defaultdict

dir =Path(filedialog.askdirectory() )

output_dir = r'S:\Digital Projects\Encoding\testing\lda_corpus\ndnp'
os.chdir(output_dir)


logging.basicConfig(level=logging.INFO,format="%(asctime)s : %(levelname)s : %(message)s", datefmt='%m-%d %H:%M', filename=r'training.log', filemode='w')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO) 
formatter = logging.Formatter("%(asctime)s :%(levelname)s : %(message)s")
ch.setFormatter(formatter)
logging.getLogger('').addHandler(ch)
log1 = logging.getLogger('myapp')
logging.basicConfig(filename='training.log',format="%(asctime)s : %(levelname)s : %(message)s", filemode= 'w',  level=logging.INFO)

pdf_group = defaultdict(list)
for pdf_file in dir.rglob("*.pdf"):
    parent = pdf_file.parent
    pdf_group[parent].append(pdf_file)

for folder, pdf_list in pdf_group.items(): 
    try: 
        relative_path = folder.relative_to(dir)
        parts = relative_path.parts
        out_name = '_'.join(parts) + '.txt'
        # out_path = output_dir / out_name
        out_path = os.path.join(output_dir, out_name)

        with open(out_path, 'w', encoding='utf-8') as f: 
            for pdf_file in sorted(pdf_list): 
                try: 
                    reader = PdfReader(pdf_file)
                    page = reader.pages[0]
                    text = page.extract_text() or ""
                    if text.strip(): 
                        f.write(f'{text} \n')
                        logging.info(f'[SUCCESS]: Extracted {pdf_file}')
                except Exception as e: 
                        logging.info(f'[FAILED]: {pdf_file}: {e}')

            logging.info(f'[COMBINED] folder {folder} -> {out_path}')
    except Exception as e: 
         logging.info(f'[FAILED COMBINE] {folder}: {e}')