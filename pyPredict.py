#  -*- coding: utf-8 -*-
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
import tweepy
import sys
import os
import nltk 
import re
import numpy as np
import string
from unidecode import unidecode
import csv
from itertools import islice
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

#ckey='YvQZ4wcEhYo4F4Lj8PQqj500d'
#csecret='nDMvzbvQxt2zW9GZOalnOikeDcwGwPwvi2USOD5phJCmw5e9wD'
#atoken='994417184322433025-anaAEWzZErK5h8CGTxEgJMPFwfRoMNA'
#asecret='M2RwOR7CaWHMO7d83iWuKhHmEtgyaomobBNt9bsJhSF3i'

tmp = 0
ckey='COVV87zJN2wHRfAz7zB5p2QPQ'
csecret='HhNmLOsIui0rG04XltDSHdmEBNf9IAtkZeW17U7pFYAuKf8qiv'
atoken='2205031009-a7FMWRzzTi5wooSMFkYUqyiq1aGREBSMCyBX2vw'
asecret='TfPQ5V8X9BOwjWQU7UBTJHTR8kYJzyhM1em8I4YGIcZxh'
auth=tweepy.OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
api=tweepy.API(auth)

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


model_IE = pickle.load(open("Pickle_Data/BNIEFinal.sav", 'rb'))
model_SN = pickle.load(open("Pickle_Data/BNSNFinal.sav", 'rb'))
model_TF = pickle.load(open('Pickle_Data/BNTFFinal.sav', 'rb'))
model_PJ = pickle.load(open('Pickle_Data/BNPJFinal.sav', 'rb'))

def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
def preproc(s):
	#s=emoji_pattern.sub(r'', s) # no emoji
	s= unidecode(s)
	POSTagger=preprocess(s)
	#print(POSTagger)

	tweet=' '.join(POSTagger)
	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(tweet)
	filtered_sentence = [w for w in word_tokens if not w in stop_words]
	'''filtered_sentence = []
	for w in word_tokens:
	    if w not in stop_words:
	        filtered_sentence.append(w)'''
	#print(word_tokens)
	#print(filtered_sentence)
	stemmed_sentence=[]
	stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
	for w in filtered_sentence:
		stemmed_sentence.append(stemmer2.stem(w))
	#print(stemmed_sentence)

	temp = ' '.join(c for c in stemmed_sentence if c not in string.punctuation) 
	preProcessed=temp.split(" ")
	final=[]
	for i in preProcessed:
		if i not in final:
			if i.isdigit():
				pass
			else:
				if 'http' not in i:
					final.append(i)
	temp1=' '.join(c for c in final)
	#print(preProcessed)
	return temp1

def getTweets(tweets):
	csvFile = open('user.csv', 'a', newline='')
	csvWriter = csv.writer(csvFile)
	try:
		for i in range(0,4):
			#tweets=api.user_timeline(screen_name = user, count = 1000, include_rts=True, page=i)
			for status in tweets:
				tw=preproc(status.text)
				#print(tw)
				if tw.find(" ") == -1:
					tw="blank"
				csvWriter.writerow([tw])
	except tweepy.TweepError:
		print("Failed to run the command on that user, Skipping...")
	csvFile.close()


def runPredictor(tweets):
	#username=input("Please Enter Twitter Account handle: ")
	getTweets(tweets)
	with open('user.csv','rt') as f:
		csvReader=csv.reader(f)
		tweetList=[rows[0] for rows in csvReader]
	os.remove('user.csv')
	with open('CSV_Data/newfrequency300.csv','rt') as f:
		csvReader=csv.reader(f)
		mydict={rows[1]: int(rows[0]) for rows in csvReader}

	vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
	x=vectorizer.fit_transform(tweetList).toarray()
	df=pd.DataFrame(x)


	

	answer=[]
	IE=model_IE.predict(df)
	SN=model_SN.predict(df)
	TF=model_TF.predict(df)
	PJ=model_PJ.predict(df)


	b = Counter(IE)
	value=b.most_common(1)
	#print(value)
	if value[0][0] == 1.0:
		answer.append("I")
	else:
		answer.append("E")

	b = Counter(SN)
	value=b.most_common(1)
	#print(value)
	if value[0][0] == 1.0:
		answer.append("S")
	else:
		answer.append("N")

	b = Counter(TF)
	value=b.most_common(1)
	#print(value)
	if value[0][0] == 1:
		answer.append("T")
	else:
		answer.append("F")

	b = Counter(PJ)
	value=b.most_common(1)
	#print(value)
	if value[0][0] == 1:
		answer.append("P")
	else:
		answer.append("J")
	mbti="".join(answer)
	# Classifying Personality's ==========================================>

	if mbti == 'ENFJ':
		str1 = '" The Giver "'
		#print(mbti +' - '+ str1)
		return 1
		
	elif mbti == 'ISTJ':
		str1 = '" The Inspector "'
		#print(mbti +' - '+ str1)
		return 2

	elif mbti == 'INFJ':
		str1 = '" The Counselor "'
		#print(mbti +' - '+ str1)
		return 3

	elif mbti == 'INTJ':
		str1 = '" The Mastermind "'
		#print(mbti +' - '+ str1)
		return 4

	elif mbti == 'ISTP':
		str1 = '" The Craftsman "'
		#print(mbti +' - '+ str1)
		return 5

	elif mbti == 'ESFJ':
		str1 = '" The Provider "'
		#print(mbti +' - '+ str1)
		return 6

	elif mbti == 'INFP':
		str1 = '" The Idealist "'
		#print(mbti +' - '+ str1)
		return 7

	elif mbti == 'ESFP':
		str1 = '" The Performer "'
		#print(mbti +' - '+ str1)
		return 8

	elif mbti == 'ENFP':
		str1 = '" The Champion "'
		#print(mbti +' - '+ str1)
		return 9

	elif mbti == 'ESTP':
		str1 = '" The Doer "'
		#print(mbti +' - '+ str1)
		return 10

	elif mbti == 'ESTJ':
		str1 = '" The Supervisor "'
		#print(mbti +' - '+ str1)
		return 11

	elif mbti == 'ENTJ':
		str1 = '" The Commander "'
		#print(mbti +' - '+ str1)
		return 12

	elif mbti == 'INTP':
		str1 = '" The Thinker "'
		#print(mbti +' - '+ str1)
		return 13

	elif mbti == 'ISFJ':
		str1 = '" The Nurturer "'
		#print(mbti +' - '+ str1)
		return 14

	elif mbti == 'ENTP':
		str1 = '" The Visionary "'
		#print(mbti +' - '+ str1)
		return 15

	else :
		str1 = '" The Composer "'
		#print(mbti +' - '+ str1)
		return 16

	print('===============================================================================>')


#runPredictor()