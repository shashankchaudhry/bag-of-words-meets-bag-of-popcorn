# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:01:22 2015

@author: schaud7
"""

import re
from bs4 import BeautifulSoup

class ExtractScore(object):
    pass_grade_list = ['a+', 'a', 'a-', 'b+', 'b', 'b-']    
    fail_grade_list = ['c+', 'c', 'c-', 'd+', 'd', 'd-', 'e', 'f']
    alternate_star_dict = {'*':1.0 ,'**':2.0 ,'***':3.0 ,'****':4.0 ,'*****':5.0}
    
    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
            
    @staticmethod
    def getBestNum(str):
        # split on dash, if only 1 part, continue
        check_for_minus = re.split('\-',str)
        if(len(check_for_minus) > 2):
            str = check_for_minus[len(check_for_minus) - 1]
        elif(len(check_for_minus) == 2):
            if(check_for_minus[0] == ''):
                return(0.0)
            else:
                str = check_for_minus[len(check_for_minus) - 1]
        str = str.replace(',','.')
        str_tokens = re.split('\.',str)
        num_tokens = []
        for token in str_tokens:
            if(ExtractScore.is_number(token)):
                num_tokens.append(float(token))
        if(len(num_tokens) < 1 or len(num_tokens) > 2):
            return(None)
        elif(len(num_tokens) == 1):
            return(num_tokens[0])
        else:
            num2 = num_tokens[1]/10
            while(num2 > 1):
                num2 /= 10
            return(num_tokens[0] + num2)
        

    @staticmethod
    def get_score_from_string(review):
        # 1. remove html
        review_text = BeautifulSoup(review).get_text()
        
        # 2. tokenize
        words = filter(None,re.split('[ <>\\r\\n\\t;:\\\'\\\"()?!\\\]', review_text))
        
        # 3. extract first 20 and last 20 words
        
        beginning_words = words[0:20]
        words_size = len(words)
        end_words = words[(words_size - 20):words_size]
        
        # 4. collect all possible scores for last 15 words. If decision can be made, return decision
        # if decision cant be made, return None, if no indicators, try first 15 words
        for word_list in [end_words, beginning_words]:
            indicator_list = []
            
            # a. see if there are any grades:
            for i in xrange(len(word_list) - 1):
                if(word_list[i] == 'grade'):
                    if(word_list[i+1] in ExtractScore.pass_grade_list):
                        indicator_list.append(1.0)
                    elif(word_list[i+1] in ExtractScore.fail_grade_list):
                        indicator_list.append(0.0)
                        
            # b. see if we can split any scores on /
            for i in xrange(len(word_list)):
                pair = word_list[i].split('/')
                if(len(pair) == 2):
                    num1 = ExtractScore.getBestNum(pair[0])
                    num2 = ExtractScore.getBestNum(pair[1])
                if(len(pair) == 2 and num1 != None and num2 != None):
                    lower_num = num2
                    if(lower_num == 5.0 or lower_num == 10.0):
                        ratio = num1 / num2
                        if(ratio >= 0.7):
                            indicator_list.append(1.0)
                        elif(ratio < 0.6):
                            indicator_list.append(0.0)
                        else:
                            indicator_list.append(0.5)
                            
            # c. see if we can see the phrase "out of" or "stars out of"
            for i in xrange(1, len(word_list)-2):
                if(word_list[i] == 'out' and word_list[i+1] == 'of'):
                    if(word_list[i-1] == 'stars'):
                        if((i-2) >= 0 and (i+2) < len(word_list) and word_list[i-2] in ExtractScore.alternate_star_dict.keys() and \
                            word_list[i+2] in ExtractScore.alternate_star_dict.keys()):
                            ratio = ExtractScore.alternate_star_dict[word_list[i-2]]/ExtractScore.alternate_star_dict[word_list[i+2]]
                            if(ratio >= 0.7):
                                indicator_list.append(1.0)
                            elif(ratio < 0.6):
                                indicator_list.append(0.0)
                            else:
                                indicator_list.append(0.5)
                        elif((i-2) >= 0 and (i+2) < len(word_list) and ExtractScore.getBestNum(word_list[i-2]) != None and \
                            ExtractScore.getBestNum(word_list[i+2]) != None):
                            ratio = ExtractScore.getBestNum(word_list[i-2]) / ExtractScore.getBestNum(word_list[i+2])
                            if(ratio >= 0.7):
                                indicator_list.append(1.0)
                            elif(ratio < 0.6):
                                indicator_list.append(0.0)
                            else:
                                indicator_list.append(0.5)
                    else:
                        if((i-1) >= 0 and (i+2) < len(word_list) and word_list[i-1] in ExtractScore.alternate_star_dict.keys() and \
                            word_list[i+2] in ExtractScore.alternate_star_dict.keys()):
                            ratio = ExtractScore.alternate_star_dict[word_list[i-1]]/ExtractScore.alternate_star_dict[word_list[i+2]]
                            if(ratio >= 0.7):
                                indicator_list.append(1.0)
                            elif(ratio < 0.6):
                                indicator_list.append(0.0)
                            else:
                                indicator_list.append(0.5)
                        elif((i-1) >= 0 and (i+2) < len(word_list) and ExtractScore.getBestNum(word_list[i-1]) != None and \
                            ExtractScore.getBestNum(word_list[i+2]) != None):
                            ratio = ExtractScore.getBestNum(word_list[i-1]) / ExtractScore.getBestNum(word_list[i+2])
                            if(ratio >= 0.7):
                                indicator_list.append(1.0)
                            elif(ratio < 0.6):
                                indicator_list.append(0.0)
                            else:
                                indicator_list.append(0.5)
            
            # d. return based on indicator list
            pos = 0
            neg = 0
            for ind in indicator_list:
                if(ind == 0.5):
                    return(None)
                if(ind == 1.0):
                    pos += 1
                if(ind == 0.0):
                    neg += 1
            if(pos > 0 and neg > 0):
                return(None)
            elif(pos == 0 and neg > 0):
                return(0.0)
            elif(pos > 0 and neg == 0):
                return(1.0)
                            
        return(None)
        