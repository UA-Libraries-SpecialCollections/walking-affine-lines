#!/usr/bin/python

# The file_splitter.py script splits the large names.madsrdf.xml name authority file into smaller portions
# and formats those pieces into well formed xml stub files for xml parsing

file_path = r"C:\encoding\subjects.madsrdf.xml\subjects.madsrdf.xml"
outfile_base = r'C:\encoding\subjects.madsrdf.xml'
count = 0
file_increment = 0
nl = '\n'

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        if count == 0:
            file_increment = file_increment + 1
            batch_filename = outfile_base + "\\" + str(file_increment) + "subjects.madsrdf.xml"
            outfile = open(batch_filename, 'a', encoding='utf-8')
            outfile.write(f'<?xml version="1.0" encoding="UTF-8"?>{nl}')
            outfile.write(f'<xml>{nl}')
        if count < 500000:
            count = count + 1
            line = line.removeprefix(r'<?xml version="1.0" encoding="UTF-8"?> ')
            outfile.write(line)
        if count == 500000:
            outfile.write(f'</xml>{nl}')
            outfile.close()
            count = 0

    outfile.write(f'</xml>{nl}')
    outfile.close()