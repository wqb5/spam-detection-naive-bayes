import glob
import os
import nltk
from nltk.tokenize import word_tokenize
import math # Just to implement log
import time # Just to know how much time it takes

# Loading files
def load_data(directory):
    x = []
    y = []
    for f in glob.glob(os.path.join(directory,"HAM.*.txt")):
        with open( f, 'r')as file:
            x.append(file.read())
            y.append(0) # HAM
    for f in glob.glob(os.path.join(directory,"SPAM.*.txt")):
        with open( f, 'r')as file:
            x.append(file.read())
            y.append(1) # SPAM
    return x,y

# Part A
def nb_train(x, y):
    ham_count = 0 
    spam_count = 0
    ham_fd = {} 
    spam_fd = {}
    for doc, label in zip(x,y): 
        words = word_tokenize(doc.lower())
        if label == 0:
            ham_count += 1
            for word in words:
                if word not in ham_fd:
                    ham_fd[word] = 0
                ham_fd[word] += 1
        else:
            spam_count += 1
            for word in words:
                if word not in spam_fd:
                    spam_fd[word] = 0
                spam_fd[word] += 1
    model = {'ham_count':ham_count, 'spam_count':spam_count, 'ham_fd':ham_fd, 'spam_fd':spam_fd}
    return model

# Part B
def nb_test(docs, trained_model, use_log=False, smoothing=False):
    predictions = []
    ham_count = trained_model['ham_count']
    spam_count = trained_model['spam_count']
    ham_fd = trained_model['ham_fd']
    spam_fd = trained_model['spam_fd']
    vocabulary = set(list(ham_fd.keys())+list(spam_fd.keys()))
    vocab_size = len(vocabulary)
    total_docs = ham_count + spam_count
    p_ham = ham_count/total_docs
    p_spam = spam_count/total_docs
    for doc in docs:
        words = word_tokenize(doc.lower())
        if use_log:
            ham_score = math.log(p_ham)
            spam_score = math.log(p_spam)
        else:
            ham_score = p_ham
            spam_score = p_spam
        ham_total = sum(ham_fd.values())
        spam_total = sum(spam_fd.values())
        for word in words:
            ham_word_count = ham_fd.get(word,0)
            spam_word_count = spam_fd.get(word,0)
            if smoothing:
                p_word_ham = (ham_word_count+1)/(ham_total+vocab_size)
                p_word_spam = (spam_word_count+1)/(spam_total+vocab_size)
            else:
                if ham_total > 0:
                    p_word_ham = ham_word_count/ham_total
                else:
                    p_word_ham = 0
                if spam_total > 0:
                    p_word_spam = spam_word_count/spam_total
                else:
                    p_word_spam = 0
            if use_log:
                if p_word_ham > 0:
                    ham_score += math.log(p_word_ham)
                else:
                    ham_score = 0
                if p_word_spam > 0:
                    spam_score += math.log(p_word_spam)
                else:
                    spam_score = 0
            else:
                if p_word_ham > 0:
                    ham_score *= p_word_ham
                else:
                    ham_score = 1
                if p_word_spam > 0:
                    spam_score *= p_word_spam
                else:
                    spam_score = 1
        if spam_score > ham_score:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions
        
# Part C
def f_score(y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    for true, pred in zip(y_true,y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
    if (tp+fp) != 0:
        precision = tp/(tp+fp)
    else:
        precision = 0
    if (tp+fn) != 0:
        recall = tp/(tp+fn)
    else:
        recall = 0
    if (precision+recall) != 0:
        f_score = (2*precision*recall) / (precision+recall)
    else:
        return 0
    return f_score   

x_train, y_train = load_data("./SPAM_training_set/")
start_train = time.time()
model = nb_train(x_train, y_train)
end_train = time.time()
print(f"Training time: {end_train - start_train:.2f} seconds")
x_test, y_test = load_data("./SPAM_test_set/")
y_pred1 = nb_test(x_test, model, use_log=True, smoothing=True)
y_pred2 = nb_test(x_test, model, use_log=True, smoothing=False)
y_pred3 = nb_test(x_test, model, use_log=False, smoothing=True)
y_pred4 = nb_test(x_test, model, use_log=False, smoothing=False)
print("Log&Smoothing")
print("True&True F1-Score: "+str(f_score(y_test, y_pred1)))
print("True&False F1-Score: "+str(f_score(y_test, y_pred2)))
print("False&True F1-Score: "+str(f_score(y_test, y_pred3)))
print("False&False F1-Score: "+str(f_score(y_test, y_pred4)))

# No helper functions needed ...