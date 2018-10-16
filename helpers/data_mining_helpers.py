import nltk

"""
Helper functions for data mining lab session 2017 Fall

Notations:
d - document
D - documents
V - vowels
w - word
W - words
l - letter
"""

def format_rows(docs):
    """ format the text field and strip special characters """
    D = []
    for d in docs.data:
        temp_d = " ".join(d.split("\n")).strip('\n\t')
        D.append([temp_d])
    return D

def format_labels(target, docs):
    """ format the labels """
    return docs.target_names[target]

def check_missing_values(row):
    """ functions that check and verifies if there are missing values in dataframe """
    counter = 0
    for element in row:
        if element == True:
            counter+=1
    return ("The amoung of missing records is: ", counter)

def tokenize_text(text, remove_stopwords=False):
    """
    Tokenize text using the nltk library
    """
    tokens = []
    for d in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(d, language='english'):
            # filters here
            tokens.append(word)
    return tokens

#helper functions for lab hw

def sentiment_data_dictionary(array):#creates a dictionary from the array of lines
    result_dictionary = {'sentences':[], 'scores':[]}
    temporal_array = [line.split("\t") for line in array if len(line.split("\t")) == 2]
    for line in temporal_array:
        result_dictionary['sentences'] += [line[0].strip("\n\t")]
        result_dictionary['scores'] += [line[1]]
    return result_dictionary
        
        
        
        
        
        
        