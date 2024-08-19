#Writing own TF-IDF
  
import pandas as pd
import numpy as np 

#Opening text file 
# with open('data.txt', 'r', encoding='utf-8') as file:
#     content = file.read()
#     #print(content)

# data = content.split("\n")

def tf_idf(texts : list):
    #Creating and fullfiling array with unique words from file
    vocab = []
    for text in texts:
        for word in text.split():
            if not word in vocab:
                vocab.append(word)

    #Writing TF function (number of this word in text / number of all words in text)
    #tf dict is a dict of {word: [tf_word_in_doc1, tf_word_in_doc2 ...]}
    tf_dict = {}
    for word in vocab:
        word_tfs = []
        for text in texts:
            if word in text:
                cnt = text.split().count(word)
                word_tfs.append(cnt / len(text.split()))
            else:
                word_tfs.append(0)    
        tf_dict[word] = word_tfs        

    #Writing IDF function
    #idf = log( number of all documents / number of all documents that have this word)
    idf_dict = {}
    for word in vocab:
        cnt = 0
        for text in texts:
            if word in text.split():
                cnt += 1
        idf_dict[word] = np.log(len(texts) / cnt)         


    #Writing TF-IDF
    tf_idf = {}
    for word in vocab:
        tf_idf[word] = np.array(tf_dict[word])*idf_dict[word]
    return tf_idf


