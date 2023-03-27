import os
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import operator
import math

# Creating tokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

# Creating Stopwords_list
Stopwords_list = stopwords.words('english')

#Creating stemmer variable
stemmer = PorterStemmer()

# initialaizing N variable for total number of documents
N = 0
document = {}
Dict_tokens_normal = {}
posting_list = {}
token_set = set()

#---------------1) Read/clean and find term frequency of each of the US president address files
corpusroot = './US_Inaugural_Addresses'
docs = []
for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close()
        doc = doc.lower()
        docs.append(doc)
        get_tokens = tokenizer.tokenize(doc)
        tokens = [stemmer.stem(token) for token in get_tokens if token not in Stopwords_list]
        token_set.update(tokens)
        term_frequency = dict(Counter(tokens))
        document[filename] = term_frequency


# Calculate Tfidf score
def getidf(token):
    document_frequency = 0 ; Tfidf_score =0
    for file in document:
        temp = document[file]
        if token in temp:
            document_frequency += 1
    if document_frequency == 0:
        return -1
    Tfidf_score = math.log10(document_frequency)
    return Tfidf_score


#calculate tf,idf and the tf matching score
def get_raw_weight(filename, token):
    if token not in document[filename]:
        return 0
    tf = 1 + math.log10( document[filename][token] )
    idf = getidf(token)
    score = tf * idf
    return score


def getweight(filename, token):
    if token not in document[filename]:
        return 0
    return document[filename][token]

def calculate_score():
    for file in document:
        l = 0
        for token in document[file]:
            score = get_raw_weight(file, token)
            document[file][token] = score
            l += score ** 2
        l = math.sqrt(l)
        Dict_tokens_normal[file] = l


# 5) normalise the tf-idf score
def get_normalized_score():
    for file in document:
        temp = Dict_tokens_normal[file]
        for token in document[file]:
            document[file][token] = document[file][token]/temp

#create posting list for all tokens
def postings_list_create():
    for token in token_set:
        tempdict = {}
        for file in document:
            if token in document[file]:
                tempdict[file] = float(document[file][token])
        tempdict = sorted(tempdict.items(), key = operator.itemgetter(1), reverse = True)
        posting_list[token] = tempdict



#calculate cosine similarity score for a given query

def query(query_string):
    query_string = query_string.lower()
    query_string = tokenizer.tokenize(query_string)
    # Tokenize
    tokens = [stemmer.stem(token) for token in query_string if token not in Stopwords_list]
    tokens = dict(Counter(tokens))
    normalizing_unit = 0

    # tf weight
    for token in tokens:
        tokens[token] = 1 + math.log10(tokens[token])
        normalizing_unit += tokens[token] ** 2
    normalizing_unit = math.sqrt(normalizing_unit)
    present = "no"
    #normalize
    for token in tokens:
        tokens[token] = tokens[token] / normalizing_unit
        if token in posting_list:
            present = "yes" 
    if present == "no":
        return ("None", 0.000000000000)
    doc_similarity = {} 
    not_top_10 = [] 
    top_10_postinglist = {}
    for token in tokens:
        if token in posting_list:
           temp = dict(posting_list[token][0:10])
           top_10_postinglist[token] = temp
    for token in top_10_postinglist:
        for filename in top_10_postinglist[token]:
            if filename not in doc_similarity:
                doc_similarity[filename] = 0
    for token in top_10_postinglist:
        for filenames in doc_similarity:
            if filenames in top_10_postinglist[token]:
                doc_similarity[filenames] += top_10_postinglist[token][filenames] * tokens[token]
            else:
                doc_similarity[filenames] += posting_list[token][0][1] * tokens[token]
                not_top_10.append(filenames)           
    doc_similarity = sorted(doc_similarity.items(), key = operator.itemgetter(1), reverse = True)
    if (doc_similarity[0][0] in not_top_10):
        return ("fetch more", 0.0)
    else:
        return (doc_similarity[0][0],doc_similarity[0][1])




calculate_score()
get_normalized_score()
postings_list_create()


print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('war'))
print("%.12f" % getidf('power'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('02_washington_1793.txt','arrive'))
print("%.12f" % getweight('07_madison_1813.txt','war'))
print("%.12f" % getweight('12_jackson_1833.txt','union'))
print("%.12f" % getweight('09_monroe_1821.txt','great'))
print("%.12f" % getweight('05_jefferson_1805.txt','public'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("false public"))
print("(%s, %.12f)" % query("people institutions"))
print("(%s, %.12f)" % query("violated willingly"))