import os 
import tkinter as tk 
from tkinter import filedialog

dir1 = tk.filedialog.askdirectory(title = 'directory 1'  )
dir2 = tk.filedialog.askdirectory(title = 'directory 2'  )

for text in os.scandir(dir1): 
    if text.is_file() and text.name.endswith('.txt'): 
        with open (text,'rb') as f:
                content = f.read()
                raw = content.decode('utf-16-le')
                if raw.startswith('\ufeff'): 
                     raw = raw[1:]
                save_path = os.path.join(dir2, text.name)
                with open(save_path, 'w', encoding='utf-8') as w: 
                    w.write(raw)
                    print(f'{text.name} is processed \n ')