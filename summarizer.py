from sklearn.datasets import load_files
import nltk
#nltk.download('stopwords')
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
#from gensim.models import Word2Vec
#from gensim.models import KeyedVectors

text1 = "It has been said that except for man, nowhere in the world is there anything to compare with the incredible efficiency of the industry of the honeybee. Inside the beehive each bee has a special job to do and the whole process runs smoothly. Bees need two different kinds of food. One is honey made from nectar, the sugary juice that collects in the heart of the flowers. The other comes from the anthers of flowers, which contain numerous small grains called pollen. Just as flowers have different colours, so do their pollen. Let us go with the honeybee from her flower to the hive and see what happens. Most bees gather only pollen or nectar. As she sucks the nectar from the flower, it is stored in her special honey stomach ready to be transferred to the honey making bees in the hive. If hungry she opens a valve in the nectar sac and a portion of the payload passes through to her own stomach to be converted to energy for her own needs. The bee is a marvelous flying machine."
text2 = "Nectar is the main ingredient for honey and also the main source of energy for bees. Using a long straw like tongue called a proboscis, honey bees suck up nectar droplets from the flower's special nectar-making organ, called the nectary. When the nectar reaches the bees honey stomach, the stomach begins to break down the complex sugars of the nectar into more simple sugars that are less prone to crystallization, or becoming solid. This process is called inversion. Once a worker honey bee returns to the colony, it passes the nectar onto another younger bee called a house bee which is between 12 to 17 days old. House bees take the nectar inside the colony and pack it away in hexagon shaped beeswax honey cells. They then turn the nectar into honey by drying it out using a warm breeze made with their wings. Once the honey has dried out, they put a lid over the honey cell using fresh beeswax kind of like a little honey jar. In the winter, when the flowers have finished blooming and there's not as much nectar available, the bees can open this lid and share the honey they saved."
all_texts = [text1, text2]

import numpy as np
import pandas as pd
import nltk
#nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from sklearn.metrics.pairwise import cosine_similarity
import re
import networkx as nx

#Textrank Implementation
df = pd.read_csv("texts.csv", encoding='utf-8')

#Tokenize sentences
from nltk.tokenize import sent_tokenize
sentences = []
for s in df['article_text']:
    sentences.append(sent_tokenize(s))
sentences = [y for x in sentences for y in x] # flatten list

#Extracting word vectors (from wikipedia + Gigaword 5 GloVe database)
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
len(word_embeddings)

#Pre-Processing ()
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
clean_sentences = [s.lower() for s in clean_sentences]
def remove_stopwords(sent):
    sent_new = " ".join([i for i in sent if i not in stop_words])
    return sent_new

clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

sentence_vectors = []
for i in clean_sentences:
    if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
        v = np.zeros((100,))
    sentence_vectors.append(v)

# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

# Extract top 10 sentences as the summary
full_sum = ""
for i in range(10):
    full_sum += ranked_sentences[i][1]
def frequency_table(data):
    '''Tokenizing text by word and building a frequency table.'''
    words = word_tokenize(data)
    stop_words = set(stopwords.words("english"))

    freqtable = {}
    for word in words:
        word = word.lower()
        if word not in stop_words:
            if word in freqtable:
                freqtable[word] += 1
            else:
                freqtable[word] = 1
   # print(freqtable.items())
    return freqtable


def scores(data, freqtable):
    '''Creates a dictionary of scores for each sentence.'''
    sentences = sent_tokenize(data)
    sent_score = {}

    for sent in sentences:
        for word, freq in freqtable.items():
            if word in sent.lower():
                if sent in sent_score:
                    sent_score[sent] += freq
                else:
                    sent_score[sent] = freq
    return sent_score

# print out setences higher than average
def summary(data, sent_score):
    '''Calculate average sentence score and print sentences at least 1.5 times greater than the average'''
    sum_scores = 0
    for sentence in sent_score:
        sum_scores += sent_score[sentence]
    average_score = sum_scores / len(sent_score)

    sentences = sent_tokenize(data)
    summary = ''
    for sentence in sentences:
        if sentence in sent_score and sent_score[sentence] > average:
            summary += " " + sentence
    return summary

# print out top 5 highest scoring sentences
def summary_five(data, sent_score):
    summary = ''
    sorted_scores = sorted(sent_score, reverse = True)
    for sent in list(sorted_scores)[0:5]:
        summary += sent
    return summary

def single_summarize(data):
    '''Combines functions to summarize a single document.'''
    freqtable = frequency_table(data)
    sent_scores = scores(data, freqtable)
    single_summary = summary_five(data, sent_scores)
    return single_summary

def multi_sum(data):
    '''Generates summaries for all documents. Takes in a list of texts'''
    summaries = []
    for d in data:
        s = single_summarize(d)
        summaries.append(s)
    return summaries

def combine(summaries):
    '''Merge summaries into a single summary. Takes in a list of summaries'''
    all_sums = ''
    for s in summaries:
        all_sums += s
    return all_sums

def sum_all(data):
    '''Generates comprehensive summary. Takes in a list of texts'''
    all_summaries = multi_sum(data)
    sum_sheet = combine(all_summaries)
    final_summary = single_summarize(sum_sheet) #can change to include further redundancy reduction
    return final_summary

print("Word frequency (WF) only:")
print()
print(sum_all(all_texts))
print()
print("Textrank only: ")
print()
print(full_sum)
print()
print("Textrank + WF score: ")
print()
print(single_summarize(full_sum))
#single_summarize(text)
