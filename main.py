#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
from textblob import TextBlob
import csv
import numpy as np
import urllib.request


# In[2]:


df_input = pd.read_csv(r"D:\Projects\data-extraction-nlp\Input.xlsx - Sheet1.csv", index_col=0)
df_input


# In[3]:


list_url = []
for url in df_input['URL']:
    list_url.append(url)


# In[4]:


megaText = []
for url in list_url:
  megaText.append(requests.get(url, headers={"User-Agent": "XY"}))
    



# In[5]:


for i in range(len(megaText)):
  megaText[i] = bs(megaText[i].content)


# In[6]:


#List which contains all the text required for sentiment analysis
megaArticle = []
for i in megaText:
    article = i.find("div",{"class":"td-post-content"})
    if article is not None:
        articleContent = article.get_text()
        megaArticle.append(articleContent)
    else:
        megaArticle.append("NoneType")


# In[7]:


for i in  range(len(megaArticle)):
    print(megaArticle[i])


# In[8]:


len(megaArticle)


# In[9]:


#Writing each Article to a text file
for i in  range(len(megaArticle)):
    with open(f"inputTextFiles/{i+37}.txt", "w", encoding="utf-8") as f:
         f.writelines(megaArticle[i])


# In[10]:


personalPronouns = []
personalPronoun =['I', 'we','my', 'ours','and' 'us','My','We','Ours','Us','And']
for article in megaArticle:
    if article == "NoneType":
        personalPronouns.append(np.nan)
    else:
        ans=0
        for word in article:
            if word in personalPronoun:
                ans+=1
        personalPronouns.append(ans)

        
print(personalPronouns)


# In[11]:


#Finding the total no. of words and sentences in the article
totalWords = []
totalSentences = []

for article in megaArticle:
    if article == "NoneType":
        totalSentences.append(0)
        totalWords.append(0)
    else:
        totalSentences.append(len(sent_tokenize(article)))
        totalWords.append(len(word_tokenize(article)))

print(totalSentences)
print(totalWords)


# In[12]:


print(len(totalSentences))
print(len(totalWords))


# In[13]:


#Determining the no. of complex words 
complexWordCount = []
syllableCounts = []
for article in megaArticle:
    sylabble_count=0
    d=article.split()
    if(len(d) != 1):
        ans=0
        for word in d:
            count=0
            for i in range(len(word)):
                if(word[i]=='a' or word[i]=='e' or word[i] =='i' or word[i] == 'o' or word[i] == 'u'):
                    count+=1

                if(i==len(word)-2 and (word[i]=='e' and word[i+1]=='d')):
                    count-=1;
                if(i==len(word)-2 and (word[i]=='e' and word[i]=='s')):
                    count-=1;
            sylabble_count+=count    
            if(count>2):
                ans+=1

        syllableCounts.append(sylabble_count)
        complexWordCount.append(ans)  
    else:
        syllableCounts.append(np.nan)
        complexWordCount.append(np.nan)


# In[14]:


print(complexWordCount)
print(syllableCounts)


# In[15]:


len(complexWordCount)


# In[16]:


#Calculating the number of characters in each article


totalCharacters = []
for article in megaArticle:
    if article == "nonetype":
        totalCharacters.append(np.nan)
    else:
        characters = 0
        for word in article.split():
            characters+=len(word)
        totalCharacters.append(characters)  


# In[17]:


totalCharacters


# In[18]:


#Calculating the number of cleaned words by following the NLTK stopWords library
stopwords.words('english')


# In[19]:


megaArticleTokenized = []
for i in range(len(megaArticle)):
     megaArticleTokenized.append(word_tokenize(megaArticle[i]))


# In[20]:


megaArticleTokenized


# In[21]:


megaArticleTokenized[100]


# In[22]:


len(megaArticleTokenized)


# In[23]:


cleanedWords = []
for i in range(len(megaArticleTokenized)):
    if len(megaArticleTokenized[i]) != 1:
            cleanedWordsCount = 0;
            for word in megaArticleTokenized[i]:
                if (word not in stopwords.words('english') and word != '?' and word != '!' and word != ','):
                    cleanedWordsCount += 1 
            cleanedWords.append(cleanedWordsCount) 
    else:
        cleanedWords.append(np.nan)

cleanedWords


# In[24]:


#Trying to create the stopWord list
import os
directory = r"D:\Projects\data-extraction-nlp\stopWords"
stopWordsFile = os.listdir(directory)
print(stopWordsFile)


# In[25]:


stopWordsList = []
for file in stopWordsFile:
    f = os.path.join(directory, file)
    with open(f, 'r') as f_object:
        word_lst = f_object.readlines()
    for i in range(len(word_lst)):
        word_lst[i]= word_lst[i].replace('\n','').replace(' ','').lower()
    stopWordsList += word_lst

print(stopWordsList)


# In[26]:


stopWordsList


# In[27]:


#Creating the positive words list
with open(r"D:\Projects\data-extraction-nlp\masterDictionary\positive-words.txt", 'r') as f_object:
    positiveWords = f_object.readlines()

for i in range(len(positiveWords)):
  positiveWords[i]= positiveWords[i].replace('\n','').replace(' ','').lower()

print(positiveWords)


# In[28]:


#Creating the negative words list
with open(r"D:\Projects\data-extraction-nlp\masterDictionary\negative-words.txt", 'r') as f_object:
    negativeWords = f_object.readlines()

for i in range(len(negativeWords)):
    negativeWords[i]= negativeWords[i].replace('\n','').replace(' ','').lower()
    #negativeWords[i] = negativeWords[i].strip()

print(negativeWords)


# In[29]:


#Calculating positive and negative score while removing stopWords at the same time
positiveScore = []
negativeScore = []

for i in range(len(megaArticleTokenized)):
    if len(megaArticleTokenized[i]) != 1:
        positiveCount = 0
        negativeCount = 0
        for word in list(megaArticleTokenized[i]):
            if word in stopWordsList:
                megaArticleTokenized[i].remove(word)
            elif word in positiveWords:
                positiveCount += 1
            elif word in negativeWords:
                negativeCount += 1
        
        positiveScore.append(positiveCount)
        negativeScore.append(negativeCount)
    else:
        positiveScore.append(np.nan)
        negativeScore.append(np.nan)
        


# In[30]:


print(positiveScore)
print(negativeScore)


# In[31]:


len(positiveScore)


# In[32]:


len(negativeScore)


# In[33]:


polarityScore = []
subjectivityScore = []

#Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)
#Subjectivity Score = (Positive Score + Negative Score)/ ((Total Words after cleaning) + 0.000001)

for i in range(114):
    if positiveScore[i] == np.nan:
        polarityScore.append(np.nan)
        subjectivityScore.append(np.nan)
    else:
        polarity = (positiveScore[i] - negativeScore[i])/((positiveScore[i] + negativeScore[i]) + 0.000001)
        subjectivity = (positiveScore[i] + negativeScore[i])/(len(megaArticleTokenized[i]) + 0.000001)
        polarityScore.append(polarity)
        subjectivityScore.append(subjectivity)

print(polarityScore)
print(subjectivityScore)


# In[34]:


avgSentenceLength = []
for i in range(114):
    if(totalSentences[i] == 0):
        avgSentenceLength.append(np.nan)
    else:
        avgSentenceLength.append(totalWords[i]/totalSentences[i])


# In[35]:


print(avgSentenceLength)


# In[36]:


percentComplexWords = []
for i in range(114):
    if totalWords[i] == 0:
        percentComplexWords.append(np.nan)
    else:
        percentComplexWords.append((complexWordCount[i] / totalWords[i])*100)

print(percentComplexWords)


# In[37]:


avgWordLength = []
for i in range(114):     #Since no. of articles provided in 114
    if totalWords[i] == 0:
        avgWordLength.append(np.nan)
    
    else:
        avgWordLength.append(totalCharacters[i]/totalWords[i])


# In[38]:


df = df_input


# In[39]:


df


# In[40]:


df['POSITIVE SCORE'] = positiveScore
df['NEGATIVE SCORE'] = negativeScore
df['POLARITY SCORE'] = polarityScore
df['SUBJECTIVITY SCORE'] = subjectivityScore
df['AVG SENTENCE LENGTH'] = avgSentenceLength
df['PERCENTAGE OF COMPLEX WORDS'] = percentComplexWords
df['FOG INDEX'] = 0.4 * (df['AVG SENTENCE LENGTH'] + df['PERCENTAGE OF COMPLEX WORDS'])
df['AVG NUMBER OF WORDS PER SENTENCES'] = df['AVG SENTENCE LENGTH']
df['COMPLEX WORD COUNT'] = complexWordCount
df['WORD COUNT'] = cleanedWords
df['SYLLABLE COUNT PER WORD'] = np.array(syllableCounts)/np.array(totalWords)
df['PERSONAL PRONOUNS'] = personalPronouns
df['AVERAGE WORD LENGTH'] = avgWordLength


# In[41]:


df


# In[42]:


df.to_excel(r'D:\Projects\data-extraction-nlp\Output_Data_Strucutre.xlsx')


# In[43]:


#The required excel sheet has been created in the required folder


# In[ ]:




