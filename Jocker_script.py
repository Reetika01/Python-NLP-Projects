#!/usr/bin/env python
# coding: utf-8

# ![Joker-2019.png](attachment:Joker-2019.png)

# PLOT OF THE MOVIE JOKER:  
# This story mainly deals with what happens when a man, who is forever alone and a failure in his life, is thrown into our society. Arthur wears two masks the one he paints for his day job as a clown, and the guise he projects in a futile attempt to feel like he's part of the world around him. Isolated, bullied and disregarded by society, Fleck begins a slow descent into madness as he transforms into the criminal mastermind known as the Joker.

# About NLP:
# 
# Natural Language Processing is the practice of teaching machines to understand and interpret conversational inputs from humans. NLP based on Machine Learning can be used to establish communication channels between humans and machines. The different implementations of NLP can help businesses and individuals save time, improve efficiency and increase customer satisfaction.
# 
# Sentiment analysis uses NLP and ML to interpret and analyze emotions in subjective data like news articles and tweets. Positive, negative, and neutral opinions can be identified to determine a customer’s sentiment towards a brand, product, or service. Sentiment analysis is used to gauge public opinion, monitor brand reputation, and better understand customer experiences.
# 

# In[1]:


#open the file
text_file = open("C:/Users/HP/Downloads/joker_script.txt")


# In[2]:


#read the file
text=text_file.read()


# In[3]:


#check the datatype of the file
print (type(text))
print("\n")


# In[4]:


#print the text file
print(text)
print("\n")


# In[5]:


#to check the length of the file
print (len(text))


# # Tokenization
# Converting sentences & words into understandable bits of data that a program can work with.

# In[6]:


#import required libraries :
from nltk import sent_tokenize
from nltk import word_tokenize


# In[7]:


#tokenize the text by sentences :
sentences = sent_tokenize(text)


# In[8]:


words = word_tokenize(text)


# In[9]:


total_documents = len(sentences)
total_documents


# In[10]:


#how many words are there?
print(len(words))


# In[11]:


#get all the words in the file
print (words)


# In[12]:


#get all the sentences in the file
print (sentences)


# In[13]:


#import required libraries
from nltk.probability import FreqDist


# In[14]:


#find the frequency
fdist = FreqDist(words)


# In[15]:


#print 20 most common words
fdist.most_common(20)


# In[16]:


fdist.tabulate() #display the freq of the word in table form


# In[17]:


#plot the grapf for fdist
import matplotlib.pyplot as plt
fdist.plot(20)


# # Removing punctuaion 
# The punctuation removal process will help to treat each text equally.

# In[18]:


#removing punctuation
#empty list to store words after processing them to remove punctuation
words_no_punc = []


# In[19]:


# is alpha checks if words are composed entirely of alphabetical char
for w in words:
    if w.isalpha():
        words_no_punc.append(w.lower())
#The purpose of this loop is to clean up a list of words by removing 
#any non-alphabetical characters and converting all the words to lowercase        


# In[20]:


print (words_no_punc)
print("\n") #to add empty line to separates the output from any subsequent code output


# In[21]:


print (words_no_punc)


# In[22]:


#check the length of the words with no punctuations
print(len(words_no_punc))


# In[23]:


fdist = FreqDist(words_no_punc)
fdist.most_common(10)


# In[24]:


fdist.plot(10)


# # Stopwords: print & remove
# The most common words in any natural language which might not add much value to the meaning of the document. 

# In[25]:


#importing stopwords
from nltk.corpus import stopwords


# In[26]:


#list of stopwords
stopwords = stopwords.words("english")
print(stopwords)


# In[27]:


#empty list to store clean words
clean_words = []


# In[28]:


# append used to add new element(words, sent, number etc) at the end of list 


# In[29]:


for w in words_no_punc:
    if w not in stopwords:
        clean_words.append(w)


# In[30]:


print (clean_words) #to print all clean words
print("\n") #to add empty line for better understanding
print(len(clean_words)) #to check the length of clean words


# In[31]:


#final frequencing distribution
fdist = FreqDist(clean_words)
fdist.most_common(10)


# In[32]:


fdist.plot(20) # creating graph for common 20 freq dist


# In[33]:


#Removing stopwords
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
sw = set(stopwords.words('english'))
print("================================")
print(sw)
print("================================")
print(len(sw))
print("================================")


# # Stemming
# Stemming is the process of reducing a word to its root form (i.e., its stem) by removing its suffixes, without considering its context or part of speech

# In[34]:


import nltk
from nltk.stem import PorterStemmer #nltk module for stemming
from nltk.tokenize import word_tokenize #tokenizing the text into words

# initialize the stemmer
stemmer = PorterStemmer()

# tokenize the text into words
words = word_tokenize(text)

# stem each word in the text
stemmed_words = [stemmer.stem(word) for word in words]

# join the stemmed words into a string
stemmed_text = ' '.join(stemmed_words)

# print the stemmed text
print(stemmed_text)


# In[35]:


print(len(stemmed_text))


# # lemmatization
# Lemmatization is the process of reducing a word to its base or dictionary form (i.e., its lemma) while considering its context and part of speech.

# In[36]:


import nltk
from nltk.stem import WordNetLemmatizer #for lemmatization
from nltk.tokenize import word_tokenize

# initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

words = word_tokenize(text)
# lemmatize each word in the text
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

# join the lemmatized words into a string
lemmatized_text = ' '.join(lemmatized_words)

# print the lemmatized text
print(lemmatized_text)


# In[37]:


print(len(lemmatized_text))


# # POS tagging
# Categorizing words in a text (corpus) in correspondence with a particular part of speech, depending on the definition of the word and its context.

# In[38]:


#POS tagging of words
tags = nltk.pos_tag(words)
tags


# # Bag of words
# A statistical language model used to analyze text and documents based on word count.

# In[39]:


#sklearn or scikit-learn is a library
#countvec is tool for extracting text, convert a collection of text documents into a matrix of token counts,
from sklearn.feature_extraction.text import CountVectorizer


# In[40]:


sentences # to print sentences from doc


# In[41]:


#create an object:
cv = CountVectorizer()


# In[42]:


#generate output for bag of words:
B_O_W = cv.fit_transform(sentences).toarray()


# In[43]:


#total words with their index in model:
cv.vocabulary_


# In[44]:


#features:
cv.get_feature_names()


# In[45]:


#show the output
B_O_W


# ## Creating frequency Matrix for the tokens 
# ## Summarizing
# Text summarization is the process of breaking down lengthy text into digestible paragraphs or sentences. This method extracts vital information while also preserving the meaning of the text.

# In[46]:


def create_frequency_matrix(sentences): # creating freq matrix for list of sent
    frequency_matrix = {}
    sw = set(stopwords.words('english')) #takes eng stopwords
    ps = PorterStemmer()
    
 # now it iterates throught each sent & create freq table   
    for sent in sentences: 
        freq_table = {}
        words = word_tokenize(sent) # tokenize each sent into words
        for word in words:
            word = word.lower() # into lower case
            word = ps.stem(word) # applying stemming using porter stemmer
 # if stemmed word is stopword it skipped, if word is already in freq table, its freq will increase by 1           
            if word in sw:
                continue
            if word in freq_table:
                freq_table[word] = freq_table[word] + 1
            else:
                freq_table[word] = 1
        frequency_matrix[sent[:15]] = freq_table
    return frequency_matrix


# ## Manual Term_Frequency Computation

# In[47]:


#creating TF table for sent, it adds tf table to tf matrix corresponding
def create_tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, f_table in freq_matrix.items():
        tf_table = {}
        count_words_in_sent = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sent
        tf_matrix[sent] = tf_table
    return tf_matrix


# ## Creating a Table for document per words

# In[48]:


#Doc per word is calculating inverse document frrquency in text mining & info retrieval
def create_document_per_words(freq_matrix):
    word_per_doc_table = {}
    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] = word_per_doc_table[word] + 1
            else:
                word_per_doc_table[word] = 1
    return word_per_doc_table


# ## Manual IDF Computation

# In[49]:


# creates idf matrix from freq matrix, doc per word & total no. of doc in collection
import math
def create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}
    for sent, f_table in freq_matrix.items():
        idf_table = {}
        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))
        idf_matrix[sent] = idf_table
    return idf_matrix


# ## TF-IDF Computation

# In[50]:


# create TF-IDF matrix from tf & idf matrix
def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sent1] = tf_idf_table
    return tf_idf_matrix


# ## Weighting the word in sentences- Scoring

# In[51]:


# TF-IDF as input & returns a dict where key sent in TF_IDF matrix
# if sent 0+ words, function computes avg score of sent by divinding 
#total score of sent by no. of words in sent.
def score_sentences(tf_idf_matrix) -> dict: 
    sentence_val = {}
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence = total_score_per_sentence + score
        if count_words_in_sentence > 0:
            sentence_val[sent] = total_score_per_sentence / count_words_in_sentence
    return sentence_val


# ## Average sentence score- Threashold

# In[52]:


#avg score returned as integer.higher scores indicating a higher overall importance.
def find_average_score(sentence_val) -> int:
    sum_values = 0
    for entry in sentence_val:
        sum_values = sum_values + sentence_val[entry]
    average = sum_values / len(sentence_val)
    return average


# ## Generating Summary

# In[53]:


#sed to generate a summary of the original text by selecting the most 
#important sentences based on their scores. The threshold parameter can be 
#adjusted to control the length and level of detail of the summary.

def generate_summary(sentences, sentence_val, threshold):
    sentence_count = 0
    summary = ""
    for sentence in sentences:
        if sentence[:15] in sentence_val and sentence_val[sentence[:15]] >= (threshold):
            summary = summary + " " + sentence
            sentence_count = sentence_count + 1
    return summary


# ## call everything and get the summarization done

# In[54]:


# generate summary based on TF_IDf algorithm.
freq_matrix = create_frequency_matrix(sentences)
tf_matrix = create_tf_matrix(freq_matrix)
count_doc_per_words = create_document_per_words(freq_matrix)
idf_matrix = create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)
sentence_scores = score_sentences(tf_idf_matrix)
threshold = find_average_score(sentence_scores)
summary = generate_summary(sentences, sentence_scores, threshold)
summary


# # word cloud
# Word clouds or tag clouds are graphical representations of word frequency that give greater prominence to words that appear more frequently in a source text.

# In[55]:


import nltk
from nltk.corpus import reuters 


# In[56]:


#importing imp. library for word cloud
from wordcloud import WordCloud 
import matplotlib.pyplot as plt #visualizing/for graph & chrts
from PIL import Image # Python Imaging Library for processing image
import numpy as np # numerical computing for working with arrays and matrices


# In[57]:


#import/open the image in jpeg format(copy the path)
im = np.array(Image.open("C:/Users/HP/Downloads/joker.png"))


# In[58]:


#mask image is that defines the shape of the wordcloud

wordcloud = WordCloud(mask = im).generate(text) #text which text data we wnat to visualize
plt.figure(figsize = (8, 8)) #tuple specify witdth & height of figure.
plt.imshow(wordcloud) #display wordcloud using imshow function
plt.show() #display the figure with word cloud image


# # Sentiment Analysis: positive & negative word cloud
# this technique is used to determine whether data is positive, negative or neutral words, also known as opinion mining

# In[59]:


#The VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon is a pre-built lexicon (or dictionary) of 
#sentiment-related words and phrases in English.(positive & negative)


# In[60]:


# nltk.download('vader_lexicon')


# In[61]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create an instance of the SentimentIntensityAnalyzer class
sentiment_analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis using the SentimentIntensityAnalyzer object
sentiments = [(word, sentiment_analyzer.polarity_scores(word)['compound']) for word in words]


# In[62]:


sentiment_analyzer = SentimentIntensityAnalyzer() #analyzes the sentiment of a given text
sentiments = [(word, sentiment_analyzer.polarity_scores(word)['compound']) for word in words]

#Polarity: which returns a dictionary containing scores for the positive, negative, neutral, and compound sentiments of the word


# In[63]:


#creates two list of words: (+ -)based on sentiment scores of words calculated using vader

from wordcloud import WordCloud

positive_words = [word for word, sentiment in sentiments if sentiment > 0]
negative_words = [word for word, sentiment in sentiments if sentiment < 0]

positive_wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',
                min_font_size = 10).generate(" ".join(positive_words))


positive_wordcloud


# In[64]:


# to specify the height, font, colour of the word cloud


# In[65]:


negative_wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(" ".join(negative_words))

negative_wordcloud


# In[66]:


#creating poitive words
import matplotlib.pyplot as plt

plt.figure(figsize = (3, 3), facecolor = None) 
plt.imshow(positive_wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)


# In[67]:


#creating negative words
plt.figure(figsize = (3, 3), facecolor = None) 
plt.imshow(negative_wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)


# In[68]:


# to create a word cloud of positive words
from wordcloud import WordCloud
from PIL import Image
import numpy as np

# Load the image mask
mask = np.array(Image.open("C:/Users/HP/Downloads/Jokker1.png"))

# Create a WordCloud object with the mask
positive_wordcloud = WordCloud(width=800, height=800, background_color='black', mask=mask)

# Generate the word cloud from the positive words
positive_wordcloud.generate_from_text(" ".join(positive_words))

# Display the word cloud image
import matplotlib.pyplot as plt
plt.imshow(positive_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.show()


# In[69]:


# to create a word cloud of negative words

from wordcloud import WordCloud
from PIL import Image
import numpy as np

# Load the image mask
mask = np.array(Image.open("C:/Users/HP/Downloads/J0O.png"))

# Create a WordCloud object with the mask
negative_wordcloud = WordCloud(width=800, height=800, background_color='red', mask=mask, collocations=False)

# Generate the word cloud from the positive words
negative_wordcloud.generate_from_text(" ".join(negative_words))

# Display the word cloud image
import matplotlib.pyplot as plt
plt.imshow(negative_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.show()


# # Clustering : Scattered Plot
# Analysing how the sentences from different paragraph are interrelated, and on the bases of it how the cluster formed

# In[70]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Define the documents
docs = [
    "I used to think that my life was a tragedy, but now I realize, it's a comedy.",
    "I hope my death makes more cents than my life.",
    "Is it just me, or is it getting crazier out there?",
    "The worst part of having a mental illness is people expect you to behave as if you don't."
]

# Convert the documents into a matrix of TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(docs)

# Perform k-means clustering with k=2
k = 2
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

# Reduce the dimensionality of the feature matrix to 2 dimensions for plotting
pca = PCA(n_components=2).fit(X.toarray())
data2D = pca.transform(X.toarray())

# Plot the clusters
plt.figure(figsize=(5, 6))
for i in range(k):
    points = np.array([data2D[j] for j in range(len(docs)) if model.labels_[j] == i])
    plt.scatter(points[:, 0], points[:, 1], s=30, label=f'Cluster {i+1}')
plt.legend()
plt.title(f'K-means clustering of {len(docs)} documents')
plt.show()


# # Translation of text
# From English to Korean & Hindi

# In[71]:


# installing library: Googletrans library, which is a free and open-source 
#library that allows us to translate text from one language to another.


# In[72]:


#pip install googletrans==4.0.0-rc1


# In[73]:


# Translation from english to korean
from googletrans import Translator

translator = Translator()
sentence ="The worst part of having a mental illness is people expect you to behave as if you don't."
translated = translator.translate(sentence, src='en', dest='ko')

print(translated.text)


# In[75]:


# Translation from english to hindi
from googletrans import Translator

translator = Translator()
sentence ="The worst part of having a mental illness is people expect you to behave as if you don't."
translated = translator.translate(sentence, src='en', dest='hindi')

print(translated.text)


# # CONCLUSION

# In this project, first we import, open, read & print the script of Joker movie in a text form. to begin with, I first check the type of data this file contains i.e string type, and how long the file is. Next step is to tokenize the content on the basis of words and sentences using sent_tokenize & word_tokenize.(importing functions from Nltk library).
#     Now, let's find the frequency  of the most common words in the file, from which, I have imported the FreqDist function. With the help of matplotlib,I have created the graph of the total words & sentences. After this remove unnecessary words,i.e removing punctuation and make the plot from it. To increase the search performance, I have removed stopwords i.e remove the low-level information from our text in order to give more focus to the important information.
#     Next step is to be done with Stemming and Lemmatization, so that it reduced those words which are not important and make our data easy to work with. Now, I will categorize the words in a text (corpus) in correspondence with a particular part of speech, i.e POS tagging and analyse text document based on word count (Bag of Words).
#     Now to summarize the data I will be creating frequency Matrix for the tokens. Making of word cloud on the basis of sentiment analysis of words having positive, negative or netural meaning. For that I have imported few images in which shape I want to make my word cloud.
#     I wanted to check how the data of each paragraph is interrelated with each other, and on the basis of that data, I have done clustering using scatterd plot. This will show which data belongs to which cluster and how they are correspondence with each other. For clustering, we need to import libraries like pandas, numpy, matplotlib, kmeans, TfidfVectorizer.
#     To get the data in any other language we can translate the data into that language. To check this I have translate one sentence into Korean and Hindi, for which I have to install Googletrans library and from it importing the translator. 
#     

# In[ ]:




