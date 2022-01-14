# import munkres
import numpy as np
import collections
import time
# !/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn
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
from pickle_picky import *

# In[71]:


"""
输入格式要求：
csv列的顺序：
性别 in ('男','女') ||  身高(int)  || 期望的身高  |'all'|int|int-int  || 年级(此项暂时没有被考虑) ||  自我介绍   ||对对方的期望 || 校区 0|1|2

可调参数：
self.get_Matrix()方法中的 weight_dis,weight_height  分别表示 校区距离 以及 身高与期望的差距 的得分权重

"""

bert_model = SentenceTransformer('bert-base-chinese')


class SCCL_BERT(nn.Module):
    '''
    SCCL_PCNN_encoder.train 是需要的
    '''

    def __init__(self, bert_model, max_length, device, special_tokens: list = None, open_bert=False):
        super(SCCL_BERT, self).__init__()
        print("SCCL_BERT init")
        self.device = device
        # self.pcnn = pcnn_encoder
        self.max_length = max_length
        self.open_bert = open_bert

        self.tokenizer = bert_model[0].tokenizer
        self.sentbert = bert_model[0].auto_model
        # add special tokens, 如果有需要
        if special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.sentbert.resize_token_embeddings(len(self.tokenizer))  # enlarge vocab

        self.embed_dim = self.sentbert.config.hidden_size
        # 如果不放开bert的话就冻结参数
        if open_bert == False:
            for param in self.sentbert.parameters():
                param.requires_grad = False
            self.sentbert.eval()

    def get_embeddings(self, text_arr):
        """
        把一个句子转换成一个向量
        :param text_arr: e.g.  ['我喜欢吃饭','我喜欢跑步',......]
        :return: 每一个句子的向量表示,  768 维度
        """
        # 这里的x都是文本
        feat_text = self.tokenizer.batch_encode_plus(text_arr,
                                                     max_length=self.max_length + 2,
                                                     return_tensors='pt',
                                                     padding='longest',
                                                     truncation=True)
        # feature的value都放到device中
        for k, v in feat_text.items():
            feat_text[k] = feat_text[k].to(self.device)
        bert_output = self.sentbert.forward(**feat_text)
        # 计算embedding
        attention_mask = feat_text['attention_mask'].unsqueeze(-1)
        all_output = bert_output[0]  # bert_output is a type of BaseModelOutput
        mean_output = torch.sum(all_output * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

        return mean_output





# # In[11]:
#
#
# class partner_matching:
#     def __init__(self):
#         self.stop_words = stopwords.words('english')
#         self.word_vectors = api.load("glove-wiki-gigaword-50")
#         # self.translator= Translator(from_lang="chinese",to_lang="english")
#         self.nlp = spacy.load('d:/sorftware/anaconda/lib/site-packages/en_core_web_sm/en_core_web_sm-3.1.0',
#                               disable=['parser', 'ner'])
#         self.pendix = np.array([0.0 for i in range(50)], dtype='float32')
#
#     # @staticmethod
#     def remove_stopwords(self, rev):
#         '''
#         funtion for removing stop words
#         :param rev:
#         :return:
#         '''
#         rev_new = " ".join([i for i in rev if i not in self.stop_words])
#         return rev_new
#
#     # @staticmethod
#     def lemmatization(self, texts, tags=['NOUN', 'ADJ', 'VERB']):  # filter noun and adjective
#         output = []
#         doc = self.nlp(" ".join(texts))
#         output.append(
#             [token.lemma_ for token in doc if
#              (token.pos_ in tags) and (token.lemma_ not in ['microwave', 'boy', 'girl'])])
#         return output[0]
#
#     def get_vec(self, sentence, if_print=0):
#         #         translation = BaiduTranslate(sentence)
#         # print(translation)
#         words = sentence.split(' ')
#         # print(words)
#         vec = np.array([0.0 for i in range(50)], dtype='float32')
#         words = self.remove_stopwords(words)
#         # print(words)
#         # print([words])
#         words = self.lemmatization([words])
#         if if_print: print(words)
#         for i in range(len(words)):
#             try:
#                 vec += self.word_vectors.get_vector(words[i])
#             except:
#                 vec += self.pendix
#
#         return np.array(vec)
#
#     def solution(self, text1, text2, if_print=0):
#         vec1 = self.get_vec(text1, if_print)
#         vec2 = self.get_vec(text2, if_print)
#         res = similarity(vec1, vec2)
#         return res
#

# In[2]:


def similarity(a, b,bert):
    a = bert.get_embeddings([a]).numpy().reshape(768)
    b = bert.get_embeddings([b]).numpy().reshape(768)
    a_norm = np.linalg.norm(a) + 1e-3
    b_norm = np.linalg.norm(b) + 1e-3
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos*100


def height_score(man_height: int, woman_height: list, weight_height):
    if len(woman_height) == 0:
        # 女方对身高没要求
        return 0
    elif len(woman_height) == 1:
        # 身高大于某个值
        delta_h = (man_height - woman_height[0]) * weight_height
        if delta_h < 0:
            return delta_h
        else:
            return 0
    elif len(woman_height) == 2:
        # 身高在一个区间
        if man_height >= woman_height[0] and man_height <= woman_height[1]:
            return 0
        elif man_height < woman_height[0]:
            return (man_height - woman_height[0]) * weight_height
        elif man_height > woman_height[1]:
            return -1 * (man_height - woman_height[0]) * weight_height
        else:
            print('ERROR')


# In[5]:


def get_height_range(man_ex_h_str: str):
    if man_ex_h_str.find('-') > 0:
        man_ex = [int(item) for item in man_ex_h_str.split('-')]
    elif man_ex_h_str.strip() == 'all':
        man_ex = [0, 999]
    else:
        man_ex = [int(man_ex_h_str)]
    return man_ex


# In[69]:


class cp_matcher:
    def __init__(self, embed_model, csv_path):
        self.model = embed_model
        df = pd.read_csv(csv_path, encoding='gbk', header=None)
        df_values = df.values
        sex_index = []

        man_index = 1
        woman_index = 1
        for sex in df_values[:, 0]:
            if sex == '男':
                sex_index.append(man_index)
                man_index += 1
            if sex == '女':
                sex_index.append(woman_index)
                woman_index += 1
        df['sex_index'] = sex_index

        df_values = df.values  # refash value
        man_values = []
        woman_values = []
        man_idx=[]
        woman_idx=[]
        for idx,item in enumerate(df_values):
            if item[0] == '男':
                man_values.append(item)
                man_idx.append(idx+1)
            elif item[0] == '女':
                woman_values.append(item)
                woman_idx.append(idx+1)
            else:
                print('*' * 21)
        #         man_values,woman_values = load('intro_ex.pickle')
        self.man_values = man_values
        self.woman_values = woman_values
        self.man_idx=man_idx
        self.woman_idx=woman_idx
        self.dis_M = np.array([[0, 13.32, 17.15],
                               [13.32, 0, 4.12],
                               [17.15, 4.12, 0]])
        self.n_man = len(self.man_values)
        self.n_woman = len(self.woman_values)

    def get_idx(self):
        return self.man_idx,self.woman_idx

    def get_Matrix(self, weight_dis=0.05, weight_height=0.5):
        bert = SCCL_BERT(self.model, 128, torch.device('cpu'))
        man_values = self.man_values
        woman_values = self.woman_values
        M = np.zeros([self.n_woman, self.n_man])
        error_m = []
        error_w = []
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                # soft score
                soft_score1 = similarity(man_values[j][4], woman_values[i][5],bert)
                soft_score2 = similarity(man_values[j][5], woman_values[i][4],bert)
                soft_score3 = similarity(man_values[j][4], woman_values[i][4],bert)
                soft_score = 2 * (soft_score1 + soft_score2 + soft_score3)/10
                # campus score
                campus_score = -1 * self.dis_M[int(man_values[j][6]), int(woman_values[i][6])] * weight_dis*10

                # height score
                man_height = man_values[j][1]
                man_ex_h_str = man_values[j][2]
                man_ex = get_height_range(man_ex_h_str)

                woman_height = woman_values[i][1]
                woman_ex_h_str = woman_values[i][2]
                woman_ex = get_height_range(woman_ex_h_str)

                h_score1 = height_score(man_height, woman_ex, weight_height)
                h_score2 = height_score(woman_height, man_ex, weight_height)
                h_score = (h_score1 + h_score2) * 0.5*10

                print(soft_score,campus_score,h_score)
                M[i][j] = soft_score + campus_score + h_score
        return M


class Hungarian():
    """
    """

    def __init__(self, input_matrix=None, is_profit_matrix=False):
        """
        输入为一个二维嵌套列表
        is_profit_matrix=False代表输入是消费矩阵（需要使消费最小化），反之则为利益矩阵（需要使利益最大化）
        """
        if input_matrix is not None:
            # 保存输入
            my_matrix = np.array(input_matrix)
            self._input_matrix = np.array(input_matrix)
            self._maxColumn = my_matrix.shape[1]
            self._maxRow = my_matrix.shape[0]

            # 本算法必须作用于方阵，如果不为方阵则填充0变为方阵
            matrix_size = max(self._maxColumn, self._maxRow)
            pad_columns = matrix_size - self._maxRow
            pad_rows = matrix_size - self._maxColumn
            my_matrix = np.pad(my_matrix, ((0, pad_columns), (0, pad_rows)), 'constant', constant_values=(0))
            self._input_matrix = my_matrix
            # 如果需要，则转化为消费矩阵
            if is_profit_matrix:
                my_matrix = self.make_cost_matrix(my_matrix)
            print("-" * 80)
            print("The my_matrix:")
            print(my_matrix)
            print("-" * 80)
            self._cost_matrix = my_matrix
            self._size = len(my_matrix)
            self._shape = my_matrix.shape

            # 存放算法结果
            self._results = []
            self._totalPotential = 0
        else:
            self._cost_matrix = None

    def make_cost_matrix(self, profit_matrix):
        '''利益矩阵转化为消费矩阵，输出为numpy矩阵'''
        # 消费矩阵 = 利益矩阵最大值组成的矩阵 - 利益矩阵
        matrix_shape = profit_matrix.shape
        offset_matrix = np.ones(matrix_shape, dtype=int) * profit_matrix.max()
        cost_matrix = offset_matrix - profit_matrix
        return cost_matrix

    def get_results(self):
        """返回匹配结果，包括匹配空值 """
        return self._results

    def calculate(self):
        """
        实施匈牙利算法的函数
        """
        result_matrix = self._cost_matrix.copy()

        # 步骤 1: 矩阵每一行减去本行的最小值
        for index, row in enumerate(result_matrix):
            result_matrix[index] -= row.min()

        # 步骤 2: 矩阵每一列减去本行的最小值
        for index, column in enumerate(result_matrix.T):
            result_matrix[:, index] -= column.min()
        # print('步骤2结果 ',result_matrix)
        # 步骤 3： 使用最少数量的划线覆盖矩阵中所有的0元素
        # 如果划线总数不等于矩阵的维度需要进行矩阵调整并重复循环此步骤
        total_covered = 0
        while total_covered < self._size:
            time.sleep(1)
            # print("---------------------------------------")
            # ('total_covered: ',total_covered)
            # print('result_matrix:',result_matrix)
            # 使用最少数量的划线覆盖矩阵中所有的0元素同时记录划线数量
            cover_zeros = CoverZeros(result_matrix)
            single_zero_pos_list = cover_zeros.calculate()
            covered_rows = cover_zeros.get_covered_rows()
            covered_columns = cover_zeros.get_covered_columns()
            total_covered = len(covered_rows) + len(covered_columns)

            # 如果划线总数不等于矩阵的维度需要进行矩阵调整（需要使用未覆盖处的最小元素）
            if total_covered < self._size:
                result_matrix = self._adjust_matrix_by_min_uncovered_num(result_matrix, covered_rows, covered_columns)
        # 元组形式结果对存放到列表
        self._results = single_zero_pos_list
        # 计算总期望结果
        value = 0
        for row, column in single_zero_pos_list:
            value += self._input_matrix[row, column]
        self._totalPotential = value

    def get_total_potential(self):
        return self._totalPotential

    def _adjust_matrix_by_min_uncovered_num(self, result_matrix, covered_rows, covered_columns):
        """计算未被覆盖元素中的最小值（m）,未被覆盖元素减去最小值m,行列划线交叉处加上最小值m"""
        adjusted_matrix = result_matrix
        # 计算未被覆盖元素中的最小值（m）
        elements = []
        for row_index, row in enumerate(result_matrix):
            if row_index not in covered_rows:
                for index, element in enumerate(row):
                    if index not in covered_columns:
                        elements.append(element)
        min_uncovered_num = min(elements)
        # print('min_uncovered_num:',min_uncovered_num)
        # 未被覆盖元素减去最小值m
        for row_index, row in enumerate(result_matrix):
            if row_index not in covered_rows:
                for index, element in enumerate(row):
                    if index not in covered_columns:
                        adjusted_matrix[row_index, index] -= min_uncovered_num
        # print('未被覆盖元素减去最小值m',adjusted_matrix)

        # 行列划线交叉处加上最小值m
        for row_ in covered_rows:
            for col_ in covered_columns:
                # print((row_,col_))
                adjusted_matrix[row_, col_] += min_uncovered_num
        # print('行列划线交叉处加上最小值m',adjusted_matrix)

        return adjusted_matrix


class CoverZeros():
    """
    使用最少数量的划线覆盖矩阵中的所有零
    输入为numpy方阵
    """

    def __init__(self, matrix):
        # 找到矩阵中零的位置（输出为同维度二值矩阵，0位置为true，非0位置为false）
        self._zero_locations = (matrix == 0)
        self._zero_locations_copy = self._zero_locations.copy()
        self._shape = matrix.shape

        # 存储划线盖住的行和列
        self._covered_rows = []
        self._covered_columns = []

    def get_covered_rows(self):
        """返回覆盖行索引列表"""
        return self._covered_rows

    def get_covered_columns(self):
        """返回覆盖列索引列表"""
        return self._covered_columns

    def row_scan(self, marked_zeros):
        '''扫描矩阵每一行，找到含0元素最少的行，对任意0元素标记（独立零元素），划去标记0元素（独立零元素）所在行和列存在的0元素'''
        min_row_zero_nums = [9999999, -1]
        for index, row in enumerate(self._zero_locations_copy):  # index为行号
            row_zero_nums = collections.Counter(row)[True]
            if row_zero_nums < min_row_zero_nums[0] and row_zero_nums != 0:
                # 找最少0元素的行
                min_row_zero_nums = [row_zero_nums, index]
        # 最少0元素的行
        row_min = self._zero_locations_copy[min_row_zero_nums[1], :]
        # 找到此行中任意一个0元素的索引位置即可
        row_indices, = np.where(row_min)
        # 标记该0元素
        # print('row_min',row_min)
        marked_zeros.append((min_row_zero_nums[1], row_indices[0]))
        # 划去该0元素所在行和列存在的0元素
        # 因为被覆盖，所以把二值矩阵_zero_locations中相应的行列全部置为false
        self._zero_locations_copy[:, row_indices[0]] = np.array([False for _ in range(self._shape[0])])
        self._zero_locations_copy[min_row_zero_nums[1], :] = np.array([False for _ in range(self._shape[0])])

    def calculate(self):
        '''进行计算'''
        # 储存勾选的行和列
        ticked_row = []
        ticked_col = []
        marked_zeros = []
        # 1、试指派并标记独立零元素
        while True:
            # print('_zero_locations_copy',self._zero_locations_copy)
            # 循环直到所有零元素被处理（_zero_locations中没有true）
            if True not in self._zero_locations_copy:
                break
            self.row_scan(marked_zeros)

        # 2、无被标记0（独立零元素）的行打勾
        independent_zero_row_list = [pos[0] for pos in marked_zeros]
        ticked_row = list(set(range(self._shape[0])) - set(independent_zero_row_list))
        # 重复3,4直到不能再打勾
        TICK_FLAG = True
        while TICK_FLAG:
            # print('ticked_row:',ticked_row,'   ticked_col:',ticked_col)
            TICK_FLAG = False
            # 3、对打勾的行中所含0元素的列打勾
            for row in ticked_row:
                # 找到此行
                row_array = self._zero_locations[row, :]
                # 找到此行中0元素的索引位置
                for i in range(len(row_array)):
                    if row_array[i] == True and i not in ticked_col:
                        ticked_col.append(i)
                        TICK_FLAG = True

            # 4、对打勾的列中所含独立0元素的行打勾
            for row, col in marked_zeros:
                if col in ticked_col and row not in ticked_row:
                    ticked_row.append(row)
                    FLAG = True
        # 对打勾的列和没有打勾的行画画线
        self._covered_rows = list(set(range(self._shape[0])) - set(ticked_row))
        self._covered_columns = ticked_col

        return marked_zeros


if __name__ == '__main__':
    model = bert_model
    Model = cp_matcher(model, 'cp.csv')
    df = pd.DataFrame(np.array(Model.get_Matrix()))
    hungarian = Hungarian(df + 30)
    hungarian.calculate()
    print("Calculated value:\t", hungarian.get_total_potential())
    results=hungarian.get_results()
    print("Results:\n\t", results)
    man_list,woman_list =Model.get_idx()
    print("Match:")
    df = pd.read_csv('cp.csv')
    df=df.values

    final_list = []
    print(woman_list)
    for idx, i in enumerate(woman_list):
        for cp in results:
            if cp[1] == idx + 1:
                final_list.append([i, man_list[cp[0]]])
    # out=pd.DataFrame(final_list,columns=['sex','高','要求','年级','自己描述','要求','校区'])
    out = pd.DataFrame(final_list, columns=['女', '男'])
    out.to_csv('匹配结果.csv', encoding='utf-8')
    print("-" * 80)

