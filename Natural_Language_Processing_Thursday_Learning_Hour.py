
# coding: utf-8

# <div style="display:block">
#     <div style="width: 20%; display: inline-block; text-align: left">
#         <img src="LOGO1" style="height:75px; margin-left:0px" />
#     </div>
#     <div style="width: 59%; display: inline-block;">
#         <h1  style="text-align: center; "> <u>Natural Language Processing</h1>
#         <h6  style="text-align: center; font-size: 12px"> </h6>
#     </div>
#         <div style="width: 20%; display: inline-block; text-align: right;">
#         <img src="LOGO2" style="height:80px; margin-left:40px" />
#     </div>
#     </div>
# 
# </div>
# 
# 

# ### Import the initial set of libraries to proceed 
# 

# In[1]:


import numpy as np
import pandas as pd
import nltk
import os


# ### Importing the dataset from local 

# In[2]:


os.chdir('C:/Users/deepak.singh/Desktop')


# In[3]:


data=pd.read_csv("training_data.csv")
dataframe=data


# # <font color='red'>Data Pre-Processing </font>

# In[4]:


dataframe['cleansed_request_text'].replace('',np.nan,inplace=True)
dataframe.dropna(inplace=True)


# In[5]:


train=dataframe
dataframe.head(10)


# ### Converting the dataset to lower case
# 

# In[6]:


dataframe['cleansed_request_text']=dataframe['cleansed_request_text'].str.lower()
dataframe.head(10)


# ### Removing the stopwords and special character 

# In[7]:


## Importing the stopword and puntuation from the nltk librarries 
# Remove the stopwords and puntuation(special charatcters) from the string or documents 

from nltk.corpus import stopwords
from string import punctuation

print(stopwords.words('english'))
print(punctuation)

stopwordsSet = set(stopwords.words('english')+list(punctuation)) # setting english words and list of punctuation


# In[8]:


def stopword_remove(input_text):
        words=input_text.split()
        stop_words_removed= [word for word in words if word not in stopwordsSet]
        stop_words_text= " ".join(stop_words_removed)
        return stop_words_text


# In[9]:


dataframe['cleansed_stop_removed']=dataframe['cleansed_request_text'].apply(lambda x: stopword_remove(x))


# ### Adding external Stopwords from stopwords-json
# 

# In[10]:


stopwords_json = {"en":["hii","Hii","Hi","Hey","hi","a","It","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]}
stopwords_json_en = set(stopwords_json['en'])
stopwords_nltk_en = set(stopwords.words('english'))
stopwords_punct = set(punctuation)
# Combine the stopwords. Its a lot longer so I'm not printing it out...
stoplist_combined = set.union(stopwords_json_en, stopwords_nltk_en, stopwords_punct)

# Remove the stopwords from `single_no8`.
print(stoplist_combined)


# ### Removing the numeric digit

# In[11]:


def remove_numeric(input_text):
        numeric_removed= "".join([i for i in input_text if not i.isdigit()])
        return numeric_removed


# In[12]:


dataframe['cleansed_numeric_removed']=dataframe['cleansed_stop_removed'].apply(lambda x: remove_numeric(x))


# ### Removing the words of length<3

# In[13]:


def remove_shortwords(input_text):
        remove_shortwords= " ".join([x for x in input_text.split() if len(x)>2])
        return remove_shortwords


# In[14]:


dataframe['cleaned_request_text']=dataframe['cleansed_numeric_removed'].apply(lambda x: remove_shortwords(x))


# ### Attempt to correct the incorrect words ( It takes pretty longer to run, try it seperately)

# In[15]:


get_ipython().system(u' pip install textblob')
from textblob import TextBlob


# In[16]:


dataframe['corrected_request_text']=dataframe['cleaned_request_text'].apply(lambda x: str(TextBlob(x).correct()))


# ## Stemming and lemmatisation
#  ### Required to tokenise the word and sentence
# 

# In[17]:


from nltk.tokenize import word_tokenize, sent_tokenize 


# In[18]:


nltk.download('punkt')
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer


# In[19]:


lem = WordNetLemmatizer()
string3 = "People live close when they work closely but they are closed enough"
# third string with repeated words of same meaning, so we can stem those 
from nltk.stem.lancaster import LancasterStemmer
lst = LancasterStemmer() # initiating LancasterStemmer
words_stemm = [lst.stem(word) for word in word_tokenize(string3)] # stemming similar words 


# In[20]:


# Now trying to use porter stemming. You can try something like RegEx stemming and snowball(snowball: If language is not english)

from nltk.stem.porter import *
stemmer = PorterStemmer() # Porter stemming 

words_stemm = [stemmer.stem(word) for word in word_tokenize(string3)] # stemming similar words 
print(words_stemm) 


# ### Most of the time we avoid uing these techniques as it may distort some words and hence we can loose information, Let's try one of them as see how it works

# In[21]:


def lan_stemmed(text_list):
    #Lemmatize
    clean_words=[stemmer.stem(word) for word in word_tokenize(text_list)]
    return(clean_words)  


# In[ ]:


dataframe['stemmed_request_text']=dataframe['cleaned_request_text'].apply(lambda x: lan_stemmed(x))


# ### <font color = 'teal'>Lemmatization serves the same purpose 
# 

# In[23]:


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

words_stemm = [wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(string3)] # stemming similar words 
print(words_stemm) 


# ## POS tagging 

# In[25]:


nltk.download('averaged_perceptron_tagger')
tagged_words=nltk.pos_tag(word_tokenize(string3)) 
tagged_words


# ## Getiing only the specific words tagged to noun or adjective etc.
# 

# In[26]:


length = len(tagged_words) - 1

a = []

for i in range (0,length):
    log = (tagged_words[i][1][0] in ('NN'))
    if log == True:
      a.append(tagged_words[i][0])
    log = (tagged_words[i][1][0] in ('VBD'))
    if log == True:
      a.append(tagged_words[i][0])
print(a)


# ## Getting the lemmatised verison of the words or terms using the POS tagging 
#  

# In[27]:


wnl = WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' # if mapping isn't found, fall back to Noun.
    
# `pos_tag` takes the tokenized sentence as input, i.e. list of string,
# and returns a tuple of (word, tg), i.e. list of tuples of strings
# so we need to get the tag from the 2nd element.

walking_tagged = pos_tag(word_tokenize(string3))
print(walking_tagged)


# ## Output of what will be the output of the same string passed earlier 
# 

# In[28]:


[wnl.lemmatize(word.lower(), pos=penn2morphy(string3)) for word, tag in walking_tagged]


# # New lemmatised function can be written as 
# 

# In[29]:


from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 
    
def lemmatize_sent(text): 
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(word_tokenize(text))]

lemmatize_sent('He is walking to school')


# ## Frequency based treatment, to remove the words occuring very frequently or incorrect

# In[30]:


# Converting the dataframe into list 
doc_complete=dataframe['cleaned_request_text'].values.tolist()
doc_clean=[doc.split() for doc in doc_complete]


# In[31]:


words= [] 
for list in doc_clean: 
    for word in list: 
        words.append(word.lower())
print(words)


# ## Frequency distribution of each words
# 

# In[32]:


freq_dist=nltk.FreqDist(words)
frequency_distribution=pd.DataFrame.from_dict(freq_dist,orient='Index')
frequency_distribution=frequency_distribution.reset_index()
frequency_distribution.columns=['words','frequency']
print(frequency_distribution)


# In[33]:


# Find out the quantile of the frequency 
frequency_distribution['frequency'].dropna().quantile([0.6, .999])


# In[35]:


result = frequency_distribution.sort_values(['frequency'], ascending=False)


# ### Wrtie the dataframe to analyse better and add the words to the stop word list or put a cut off and use that as filter
# 

# In[36]:


frequency_distribution.to_csv('frequency.csv')


# In[37]:


test=frequency_distribution[(frequency_distribution.frequency <=3 ) | (frequency_distribution.frequency >500)]
wastewords=pd.DataFrame(test['words'])
wastewords


# ## Creating the dataframe to lits and applying the filter to the main list of document 

# In[38]:


word_list=wastewords['words'].values.tolist()
doc_clean=[doc.split() for doc in doc_complete if doc not in word_list]

type(doc_clean)
doc_clean


# ## Bigram generation 

# In[39]:


# Generating bigram
import gensim

bigram=gensim.models.Phrases(doc_clean)
doc_clean=[bigram[line] for line in doc_clean]


# ### <font color = 'teal'>Generating trigram
# 

# In[40]:


bigram=gensim.models.Phrases(doc_clean)
doc_clean=[bigram[line] for line in doc_clean]


# # Word Embedding 

# ### Encoding the lists
# 

# In[41]:


doc_clean_df_word = [[y.encode("utf-8") for y in z] for z in doc_clean]
print(doc_clean_df_word)


# In[42]:


type(doc_clean_df_word)
doc_clean_df_word[:2]


# ## word2Vec

# In[43]:


def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['cleaned_request_text']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus

corpus = build_corpus(dataframe)
corpus[0:2]


# # <font color = 'teal'> Word2vec

# In[92]:


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import word2vec
model = word2vec.Word2Vec(corpus)


# In[99]:


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=0)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# In[101]:


tsne_plot(model)


# ## Topic Modeling 

# ###  Conversion as document term matrix
# ### Import dictionary from gensim and craeting a dictionary for the terms 
# 

# In[425]:


from gensim.corpora import Dictionary
dictionary= Dictionary(doc_clean)


# ### Creating the document term matrix using the dictionary 
# 

# In[426]:


doc_term_matrix=[dictionary.doc2bow(doc) for doc in doc_clean]


# In[427]:


print(doc_term_matrix)


# ## Creating a LDA model

# In[428]:


LDA=gensim.models.ldamodel.LdaModel


# In[429]:


import random
random.seed(3)


# In[430]:


#create the LDA model
#Value of alpha and Beta can be altered based on density required for topics in documents or terms in the topics

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel

ldamodel = LdaModel(corpus=doc_term_matrix, num_topics=15, id2word=dictionary, alpha=0.001, iterations=50, eta=.01)


# In[431]:


# Get a feel of some of the topics with their word contribution

print(ldamodel.print_topics(num_topics=3, num_words=5))


# In[432]:


# For every document what is the probability of each topics out of 15 topics 

for x in range(0,len(doc_clean)):
    print ldamodel[dictionary.doc2bow(doc_clean[x])]


# In[434]:


# Get the coherence value for the model 

from gensim.models.coherencemodel import CoherenceModel
coherence_value = CoherenceModel(model=ldamodel, corpus=doc_term_matrix, dictionary=dictionary, coherence='u_mass').get_coherence()
print(coherence_value)


# In[435]:


#Installing the pyLDAvis for the visualisation purpose 

get_ipython().system(u'pip install pyldavis')


# In[436]:


import pyLDAvis.gensim
pyLDAvis.enable_notebook()


# In[437]:


# To get the visualisation of the size of the 
pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)


# ## Creating a LSA/LSI model

# In[438]:


Lsi= gensim.models.lsimodel.LsiModel


# In[439]:


lsimodel= Lsi(corpus=doc_term_matrix, num_topics=15, id2word=dictionary)


# In[440]:


# Get a feel of some of the topics with their word contribution

print(lsimodel.print_topics(num_topics=3, num_words=5))


# In[441]:


# For every document what is the probability of each topics out of 15 topics 

for x in range(0,len(doc_clean)):
    print lsimodel[dictionary.doc2bow(doc_clean[x])]


# In[442]:


# Get the coherence value for the model 

from gensim.models.coherencemodel import CoherenceModel
coherence_value = CoherenceModel(model=lsimodel, corpus=doc_term_matrix, dictionary=dictionary, coherence='u_mass').get_coherence()
print(coherence_value)


# ## Similarly for HDP we can proceed 

# In[443]:


## Using the package imported below

hdpmodel=gensim.models.hdpmodel.HdpModel


# In[444]:


hdpmodel= HdpModel(corpus=doc_term_matrix, id2word=dictionary,alpha=.001)


# In[445]:


coherence_value = CoherenceModel(model=hdpmodel, corpus=doc_term_matrix, dictionary=dictionary, coherence='u_mass').get_coherence()
print(coherence_value)


# ## Clustering techniques can be used here for document grouping 

# In[55]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# In[56]:


vectorizer = TfidfVectorizer(stop_words='english',use_idf=True)
model = vectorizer.fit_transform(dataframe['cleaned_request_text'].str.upper())
km = KMeans(n_clusters=50,init='k-means++',max_iter=200,n_init=1)

k=km.fit(model)
terms = vectorizer.get_feature_names()
order_centroids = km.cluster_centers_.argsort()[:,::-1]
for i in range(50):
    print("cluster of words %d:" %i)
    for ind in order_centroids[i,:10]:
        print(' %s' % terms[ind])
    print() 


# In[57]:


# In order to get the SSE 

sse={}
for k in range(3, 50):
    kmeans= KMeans(n_clusters=k,max_iter=1000).fit(model)
    dataframe["clusters"]=kmeans.labels_
    sse[k]=kmeans.inertia_


# In[60]:


get_ipython().magic(u'matplotlib inline')


# # plot the sse curve to select the value of k
# 

# In[63]:


plt.figure()
plt.plot(sse.keys(), sse.values())
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# ## Output of the cluster
# 

# In[103]:


Output=train[['cleaned_request_text','clusters']].query('clusters==22')
Output.head(20)

