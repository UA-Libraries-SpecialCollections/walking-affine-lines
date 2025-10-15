##!/usr/bin/python
# C:\Python310>python.exe "S:\Digital Projects\Encoding\testing\xml_to_xsd_validator.py"
# Project hosted @ https://github.com/UA-Libraries-SpecialCollections/walking-affine-lines
# Developed by the University of Alabama Libraries Digital Services unit
# Funded by a 2025 University of Alabama Office of Economic Development FUSE Grant
# Bryce Shiver 2025 beshiver@crimson.ua.edu

# This script allows for the validation of xml files with the overall xml schema file (XSD) used for this project.
# The script will report a log (of xml files that were processed) into the working directory, labeled as "xml_validation_log.txt"  
# Overall this will provide a VALID, INVALID, and ERROR as a result from the validation (references at bottom)
# The script iterates through directory of xml files to validate with a single xsd file

# ----------------------------------------
# Disclaimer!
# This software is provided "as-is" and without warranty of any kind, either express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. Use of this software is at the user's own risk.
# By using this software, users acknowledge that it provides access to third-party APIs, which might result in financial charges if those APIs are accessed and utilized. Users are solely responsible for any and all costs, charges, fees, or expenses incurred as a result of using, accessing, or invoking these third-party APIs through this software.
# It is the user's responsibility to read and understand the terms of service, pricing details, and any other relevant information related to third-party APIs accessed through this software. The maintainers, contributors, and creators of this software shall not be held liable for any financial charges or damages that may arise from the use or misuse of these third-party APIs.
# Users are also responsible for securing their API keys, credentials, and any other sensitive information related to these third-party services. The maintainers, contributors, and creators of this software shall not be held liable for any unauthorized access, data breaches, or other security incidents related to the use of these third-party APIs.
# By using this software, the user agrees to indemnify, defend, and hold harmless the maintainers, contributors, and creators of this software from any and all claims, damages, losses, liabilities, costs, and expenses, including legal fees and expenses, arising out of or related to their use or misuse of the software and any third-party APIs accessed through it.



import xmlschema                                                       
import os    
import logging  
from tkinter import filedialog

logging.basicConfig(level=logging.INFO,format="%(asctime)s : %(levelname)s : %(message)s", datefmt='%m-%d %H:%M', filename=r'S:\Digital Projects\Encoding\testing\logs\xml_validation_log.txt', filemode='w')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO) 
formatter = logging.Formatter("%(asctime)s :%(levelname)s : %(message)s")
ch.setFormatter(formatter)
logging.getLogger('').addHandler(ch)
log1 = logging.getLogger('myapp')
logging.basicConfig(filename='training.log',format="%(asctime)s : %(levelname)s : %(message)s", filemode= 'w',  level=logging.INFO)

xml_dir = filedialog.askdirectory(title='Select Directory for XML')        

schema_dir = filedialog.askdirectory(title='Select Directory for XSD Schema')  

xsd = xmlschema.XMLSchema(schema_dir)                               

for xml in os.scandir(xml_dir): 
        if xml.is_file() and xml.name.endswith(".xml"): 
                try: 
                        xsd.validate(xml.path)  
                        logging.info(f"[VALID] {xml.name}\n")

                except xmlschema.XMLSchemaValidationError as e:
                        logging.info(f"[INVALID] {xml.name}\n") 
                        logging.info(f"[INVALID] {e}") 

                except Exception as e: 
                        logging.info(f"[ERROR] {xml.name}\n") 
                        logging.info(f"[ERROR] {e}") 
'''
VALID = passing xml file that matches with the XSD 

INVALID = improper xml file when compared to the xsd file, in the terminal the problem will be posted and a path where the issues occurs in the xml file. This also provides a referance to the schema (XSD), why the xml failes, an instance type, and the instance of failure. This is the way to troubleshoot and fix the xml file for proper validation

ERROR = There is an error in the code and process of validation, such as file read issues or anything else (not an xml validation issues but a code processing issue)
'''