#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import gensim.downloader as api
import numpy as np
# from translate import Translator
import spacy
from nltk.corpus import stopwords
import requests
import random
import json
import time
from hashlib import md5
from tqdm import tqdm
from pickle_picky import*


# In[71]:


"""
输入格式要求：
csv列的顺序：
性别 in ('男','女') ||  身高(int)  || 期望的身高  |'all'|int|int-int  || 年级(此项暂时没有被考虑) ||  自我介绍   ||对对方的期望 || 校区 0|1|2

可调参数：
self.get_Matrix()方法中的 weight_dis,weight_height  分别表示 校区距离 以及 身高与期望的差距 的得分权重

"""


# In[11]:


class partner_matching:
    def __init__(self):
        self.stop_words = load('stopwords.pickle')
        self.word_vectors = api.load("glove-wiki-gigaword-50")
        #self.translator= Translator(from_lang="chinese",to_lang="english")
        self.nlp =spacy.load('en_core_web_sm',disable=['parser', 'ner'])
        self.pendix = np.array([0.0 for i in range(50)],dtype='float32')

    #@staticmethod
    def remove_stopwords(self,rev):
        '''
        funtion for removing stop words
        :param rev:
        :return:
        '''
        rev_new = " ".join([i for i in rev if i not in self.stop_words])
        return rev_new

    #@staticmethod
    def lemmatization(self,texts, tags=['NOUN', 'ADJ','VERB']):  # filter noun and adjective
        output = []
        doc = self.nlp(" ".join(texts))
        output.append(
            [token.lemma_ for token in doc if (token.pos_ in tags) and (token.lemma_ not in ['microwave', 'boy','girl'])])
        return output[0]

    def get_vec(self,sentence,if_print=0):
#         translation = BaiduTranslate(sentence)
        #print(translation)
        words = sentence.split(' ')
        #print(words)
        vec = np.array([0.0 for i in range(50)],dtype='float32')
        words = self.remove_stopwords(words)
        #print(words)
        # print([words])
        words = self.lemmatization([words])
        if if_print:print(words)
        for i in range(len(words)):
            try:
                vec += self.word_vectors.get_vector(words[i])
            except:
                vec += self.pendix
            
        return np.array(vec)
    def solution(self,text1,text2,if_print=0):
        vec1 = self.get_vec(text1,if_print)
        vec2 = self.get_vec(text2,if_print)
        res = similarity(vec1,vec2)
        return res


# In[2]:


def similarity(a, b):
    a_norm = np.linalg.norm(a)+1e-3
    b_norm = np.linalg.norm(b)+1e-3
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos


# In[3]:


def BaiduTranslate(words):
    # Set your own appid/appkey.
    appid = '20210516000829351'
    appkey = 'xOvZiq_aGlfel1LaSMWd'

    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    from_lang = 'zh'
    to_lang =  'en'

    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    query = words

    # Generate salt and sign
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # Show response
    #print(json.dumps(result, indent=4, ensure_ascii=False))
    #print(result['trans_result'][0]['dst'])
    #print(result)
    return result['trans_result'][0]['dst']


# In[32]:


def height_score(man_height:int,woman_height:list,weight_height):
    if len(woman_height)==0:
        #女方对身高没要求
        return 0
    elif len(woman_height)==1:
        #身高大于某个值
        delta_h = (man_height-woman_height[0])*weight_height
        if delta_h<0:
            return delta_h
        else:
            return 0
    elif len(woman_height)==2:
        #身高在一个区间
        if man_height>=woman_height[0] and man_height<=woman_height[1]:
            return 0
        elif man_height<woman_height[0]:
            return (man_height-woman_height[0])*weight_height
        elif man_height>woman_height[1]:
            return -1*(man_height-woman_height[0])*weight_height
        else:
            print('ERROR')


# In[5]:


def get_height_range(man_ex_h_str:str):
    if man_ex_h_str.find('-')>0:
        man_ex = [int(item) for item in man_ex_h_str.split('-')]
    elif man_ex_h_str.strip()=='all':
        man_ex = [0,999]
    else:
        man_ex = [int(man_ex_h_str)]
    return man_ex


# In[69]:


class cp_matcher:
    def __init__(self,embed_model,csv_path):
        self.model = embed_model
        df = pd.read_csv(csv_path,encoding='gbk',header=None)
        df_values = df.values
        sex_index = []
        
        man_index = 1
        woman_index = 1
        for sex in df_values[:,0]:
            if sex=='男':
                sex_index.append(man_index)
                man_index+=1
            if sex=='女':
                sex_index.append(woman_index)
                woman_index+=1
        df['sex_index'] = sex_index
        
        df_values = df.values #refash value
        man_values = []
        woman_values = []
        for item in tqdm(df_values):
            if item[0]=='男':
                item[4] = BaiduTranslate(item[4])
                time.sleep(1)
                item[5] = BaiduTranslate(item[5])
                time.sleep(1)
                man_values.append(item)
            elif item[0]=='女':
                item[4] = BaiduTranslate(item[4])
                time.sleep(1)
                item[5] = BaiduTranslate(item[5])
                time.sleep(1)
                woman_values.append(item)
            else:
                print('*'*21)
#         man_values,woman_values = load('intro_ex.pickle')
        self.man_values= man_values
        self.woman_values  = woman_values
        self.dis_M = np.array([[0,13.32,17.15],
                  [13.32,0,4.12],
                  [17.15,4.12,0]])
        self.n_man = len(self.man_values)
        self.n_woman = len(self.woman_values)
    def get_Matrix(self,weight_dis=0.05,weight_height = 0.5):
        man_values = self.man_values
        woman_values = self.woman_values
        M = np.zeros([self.n_woman,self.n_man])
        error_m = []
        error_w = []
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                try:
                    #soft score 
                    soft_score1 = self.model.solution(man_values[j][4],woman_values[i][5])
                    soft_score2 = self.model.solution(man_values[j][5],woman_values[i][4])
                    soft_score = 2*(soft_score1+soft_score2)
                    #campus score
                    campus_score = -1*self.dis_M[int(man_values[j][6]),int(woman_values[i][6])]*weight_dis

                    #height score
                    man_height = man_values[j][1]
                    man_ex_h_str = man_values[j][2]
                    man_ex = get_height_range(man_ex_h_str)


                    woman_height = woman_values[i][1]
                    woman_ex_h_str = woman_values[i][2]
                    woman_ex = get_height_range(woman_ex_h_str)

                    h_score1 = height_score(man_height,woman_ex,weight_height)
                    h_score2 = height_score(woman_height,man_ex,weight_height)
                    h_score = (h_score1+h_score2)*0.5

                    #print(soft_score,campus_score,h_score)
                    M[i][j] = soft_score+campus_score+h_score
                except:
                    error_m.append(man_values[j])
                    error_w.append(woman_values[i])
                    print(man_values[j],woman_values[i])
                    break
        return M


# In[52]:





# In[70]:


if __name__=='__main__':
    model = partner_matching()
    Model = cp_matcher(model,'cp.csv')
    print(Model.get_Matrix())
#     Model.get_Matrix()

