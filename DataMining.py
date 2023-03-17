# Read/clean all the presidential files
import os
corpusroot = './US_Inaugural_Addresses'

docs = {}
for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close() 
        doc = doc.lower()
        docs[filename] = doc