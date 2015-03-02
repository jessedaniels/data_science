from gensim import corpora, models, similarities, matutils
import numpy as np
from numpy.random import multinomial

n_topics = 2
alpha = 50.0 / n_topics
beta = 1.

documents = ["Human machine interface for lab abc computer applications", "A survey of user opinion of computer system response time", "The EPS user interface management system", "System and human system engineering testing of EPS", "Relation of user perceived response time to error measurement", "The generation of random binary unordered trees", "The intersection graph of paths in trees", "Graph minors IV Widths of trees and well quasi ordering", "Graph minors A survey"]

stoplist = set(['a', 'and', 'for', 'of', 'to', 'in', 'the'])


def sample_topic_given_word(word, C_wt, dtm):
    word_row = np.copy(C_wt[word,])
    # Decrement word topic matrix by 1 for entries > 0 only (otherwise you get neg values)
    C_wt[word,] = np.maximum(word_row - 1, 0)
    # Create doc x topic count matrix
    # Finds the number of times topics show up in documents containing word i (hope this is right)
    C_dt = dtm[dtm[:,word] > 0,].dot(C_wt)

    prob_word_given_topic = (C_wt[word,] + beta) / (C_wt.sum(axis=0) + n_words * beta)
    prob_topic_given_doc = (C_dt.sum(axis=0) + alpha) / (C_dt.sum() + n_topics * alpha)
    
    prob_topic = prob_word_given_topic * prob_topic_given_doc / sum(prob_word_given_topic * prob_topic_given_doc)

    # Increment word topic matrix with sample from this distribution
    C_wt[word,] = word_row + multinomial(1, prob_topic, size=1)
    
    return C_wt




# List containing a list for each document containing document's cleaned words. Could do more: extremes filtering, etc.
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

# Using a normal dictionary vs hashing one for debugging
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
n_words = len(dictionary.values())

# Document term matrix in dense form for development and debugging
# dtm is num_doc x num_terms
dtm = matutils.corpus2dense(corpus, n_words).T

# Assign each word to a random topic
C_wt = multinomial(1, [1./n_topics]*n_topics, size=n_words)

#print C_wt

for gibbs_sample_iterations in range(300):
    for r in range(C_wt.shape[0]):
        C_wt = sample_topic_given_word(r, C_wt, dtm)


print C_wt

word_topic_probs = (C_wt + beta) / (np.sum(C_wt, axis=1)[:,None] + n_words * beta)
doc_topic_probs = (dtm.dot(C_wt) + alpha) / (np.sum(dtm.dot(C_wt), axis=1)[:,None] + n_topics * alpha)

print 'Document Assignment to Topics:'
for t in range(n_topics):
    print '\nTopic %s' % t
    for i in np.argsort(-doc_topic_probs, axis=0)[:,t][:5] :
        print doc_topic_probs[i,t], texts[i]

print '\nWord Assignment to Topics:'
for t in range(n_topics):
    print '\nTopic %s' % t
    for i in np.argsort(-word_topic_probs, axis=0)[:,t][:5]:
        print word_topic_probs[i,t], dictionary.values()[i]


# TODO: Gibbs sampler burn in? How should the above change?
