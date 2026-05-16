import numpy as np

X = np.array([[1, 1],[0,1],[0,0],[0,1],[1,0]])
free = np.array([1,0,0,0,1])
win = np.array([1,1,0,1,0])
y = np.array([1,1,0,0,0])

y_spam = []
y_not_spam = []
for i in range(len(y)):
    if y[i] == 0:
        y_not_spam.append(y[i])
    else:
        y_spam.append(y[i])
y_not_spam = np.array(y_not_spam)
y_spam = np.array(y_spam)

prob_spam = y_spam.shape[0] / y.shape[0]
print("prob of spam : ",prob_spam)

prob_not_spam = y_not_spam.shape[0] / y.shape[0]
print("prob of not spam : ",prob_not_spam)

## for spam class p(free=yes| spam)
prob= 0
for i in range(len(free)):
    if free[i] == 1 and y[i] == 1:
        prob = prob+1
free_spam = prob/len(y_spam)
print("p(free=yes|spam) : ",free_spam)

##for spam class P(win=yes|spam)
p=0
for i in range(len(win)):
    if win[i] == 1 and y[i] == 1:
        p = p+1
win_spam = p/len(y_spam)
print("P(win=yes|spam) : ", win_spam)

##for not spam class P(free=yes|not spam)
p_ns=0
for i in range(len(free)):
    if free[i] == 1 and y[i] == 0:
        p_ns = p_ns+1
free_ns = p_ns/len(y_not_spam)
print("P(free=yes|not spam) : ", free_ns)

##for not spam class P(win=yes|not spam)
prob_ns=0
for i in range(len(win)):
    if win[i] == 1 and y[i] == 0:
        prob_ns = prob_ns+1
win_ns = p_ns/len(y_not_spam)
print("P(win=yes|not spam) : ", win_ns)


## P(spam|X)
prob_spam_given_X = free_spam * win_spam * prob_spam
print("P(spam|X) : ",prob_spam_given_X)

##P(Not-spam|X)
prob_not_spam_given_X = win_ns * free_ns * prob_not_spam
print("P(not spam|X) : ",prob_not_spam_given_X)

if prob_spam_given_X > prob_not_spam_given_X:
    print("SPAM")