from gensim import corpora, models, similarities, matutils
import numpy as np
from numpy.random import multinomial

n_topics = 2

alpha = 0.1 / n_topics
beta = 1.

#documents = ["Human machine interface for lab abc computer applications", "A survey of user opinion of computer system response time", "The EPS user interface management system", "System and human system engineering testing of EPS", "Relation of user perceived response time to error measurement", "The generation of random binary unordered trees", "The intersection graph of paths in trees", "Graph minors IV Widths of trees and well quasi ordering", "Graph minors A survey"]

#documents = ["Human machine interface for lab abc computer applications", "A survey of user opinion of computer system response time", "The EPS user interface management system", "System and human system engineering testing of EPS", "Relation of user perceived response time to error measurement", "The baked potato plays great music", "Tony Macalpine plays good music", "Tony plays at the baked potato often", "Can't wait for Tony new album", "Amazon should carry Tony album since it carries music", "The baked potato is a great music venue", "human machine", "tony baked", "system human"]

documents = ['stream river stream river river stream river stream river river river river stream stream bank stream', 'river river bank stream stream bank bank stream bank river stream river river bank bank bank', 'stream river river river stream river stream stream river river river bank bank bank stream river', 'river stream bank bank stream river bank bank river bank river river stream stream stream river', 'river bank bank river river stream stream river river bank river river river bank bank bank', 'stream river river stream stream river bank stream river river bank stream bank river bank bank', 'bank stream river bank river river bank river bank bank bank river river river bank bank', 'river bank river stream river river river stream bank stream river bank stream bank bank river', 'stream stream river bank stream stream river stream river bank river stream stream stream bank river', 'river river stream stream bank stream bank stream stream stream river bank river river bank bank', 'money river bank river river bank bank bank bank river river loan bank stream bank stream', 'river bank stream river money bank stream river river river bank bank river bank bank river', 'stream river stream stream river river river bank river bank bank river bank stream stream river', 'stream river stream river river bank river river bank bank bank river river stream river river', 'bank river river stream river bank river bank stream river river stream stream river bank river', 'stream river stream stream river stream river river bank stream bank stream bank stream bank stream', 'stream stream stream river loan stream river stream river river river river bank money bank money', 'stream stream river bank bank river stream bank river stream bank river loan river bank bank', 'bank bank bank river river bank bank bank stream bank river bank stream bank stream river', 'stream bank bank loan bank money stream stream loan money bank stream stream bank stream stream', 'stream bank loan bank bank bank river river stream bank stream river bank bank bank river', 'bank river stream bank stream stream bank stream bank bank stream bank river bank stream bank', 'river bank river money stream loan loan loan river bank loan loan loan bank bank bank', 'bank stream loan stream bank river stream stream river stream bank bank money loan river money', 'loan stream bank money money bank river bank river bank river money money bank bank bank', 'stream river money money bank bank money river money river bank bank stream bank bank bank', 'stream bank bank bank money bank stream bank loan stream loan money bank stream bank river', 'loan bank loan bank bank river bank bank river money river loan money loan loan stream', 'bank bank money loan loan bank river stream river bank stream loan river loan bank bank', 'bank money bank bank bank bank bank bank bank river money money stream bank money bank', 'money loan bank bank river loan loan bank loan bank bank money loan river river money', 'stream money loan loan bank loan bank money river stream bank money bank money money stream', 'money bank bank loan bank stream loan bank loan money money loan bank river loan money', 'bank river money loan money bank bank money bank bank money money bank bank loan loan', 'bank bank bank loan bank loan loan loan bank money loan loan loan loan bank river', 'bank loan loan loan bank bank loan money money money money bank money stream money loan', 'money bank bank loan loan loan loan loan loan money loan money money bank loan bank', 'loan money loan loan bank bank money bank stream loan money bank bank bank money loan', 'bank bank money bank loan money bank loan stream money bank money bank bank bank money', 'bank money loan money money bank loan loan loan river bank money bank bank bank bank', 'loan bank money bank bank money money money bank money bank loan bank bank loan money', 'loan money bank money bank loan bank bank loan bank loan bank bank loan money money', 'bank bank loan loan money loan loan bank loan loan bank money money stream bank bank', 'bank bank bank money loan loan money loan money money bank money loan money loan loan', 'loan money bank loan money money loan loan bank bank loan bank loan money loan money', 'bank money loan money loan bank bank bank bank loan loan loan bank bank loan loan', 'loan bank loan money money money loan loan money loan money loan bank money bank loan', 'money loan bank loan money bank bank loan money bank loan loan loan loan money bank', 'bank bank loan loan bank money money bank bank money loan loan bank loan bank loan', 'money money bank bank loan bank loan money loan loan loan bank bank bank bank bank']

stoplist = set(['a', 'and', 'for', 'of', 'to', 'in', 'the', "can't", "at"])


def sample_topic_given_word(i, t, d, C_wt, dtm):
    # Unpack word id and current assignment from t:
    w, z = t 
    # Decrement word topic matrix by 1 for current assignment
    C_wt[w, z] -= 1
    # Create doc x topic count matrix
    # Finds the number of times topics show up in current doc. Having dtm is very convenient, but expensive to keep around. Corpus has the same info. 
    # Extracts a row from doc x topic count matrix
    C_dt = (dtm[d,]).dot(C_wt)

    prob_word_given_topic = (C_wt[w,] + beta) / (C_wt.sum(axis=0) + n_words * beta)
    prob_topic_given_doc = (C_dt + alpha) / (C_dt.sum() + n_topics * alpha)
    
    prob_topic = prob_word_given_topic * prob_topic_given_doc / sum(prob_word_given_topic * prob_topic_given_doc)

    # Increment word topic matrix with sample from this distribution
    z_sample = int(np.where(multinomial(1, prob_topic, size=1)[0] > 0)[0])
    # Update corpus with new current z assignment for word id w
    corpus[d][i] = (w, z_sample)

    #C_wt[w, z] += 1
    C_wt[w, z_sample] += 1

#np.random.shuffle(documents)
# List containing a list for each document containing document's cleaned words. Could do more: extremes filtering, etc.
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

# Using a normal dictionary vs hashing one for debugging
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
n_words = len(dictionary.values())

# Document term matrix in dense form for development and debugging
# dtm is num_doc x num_terms
# Binarizing it, but could use weighting...
dtm = np.array(matutils.corpus2dense(corpus, n_words).T > 0, dtype=float)

# Tracks number of times a word is assigned to a topic
C_wt = np.zeros((n_words, n_topics))

# Hijacking the corpus structure. The second element is now the current topic assignment.
# Track current assignments of all tokens
# Randomly initialize each token with a random topic
for doc in corpus: 
    for i, t in enumerate(doc):
        word_id = t[0]
        # Second element of term tuple is the random z assignment
        doc[i] = word_id, int(np.where(multinomial(1, [1./n_topics]*n_topics, size=1)[0] > 0)[0])
        # Increment count in C_wt for each time word is assigned to a topic
        C_wt[doc[i][0], doc[i][1]] += 1.

for gibbs_sample_iterations in range(100):
    # (doc_index, document)
    for d, doc in enumerate(corpus):
        # (word_index, (word_id, current_assignment)
        for i, t in enumerate(doc):
            sample_topic_given_word(i, t, d, C_wt, dtm)


print C_wt

word_topic_probs = (C_wt + beta) / (np.sum(C_wt, axis=0) + n_words * beta)
doc_topic_probs = (dtm.dot(C_wt) + alpha) / (np.sum(dtm.dot(C_wt), axis=1)[:,None] + n_topics * alpha)

print 'Document Assignment to Topics:'
for t in range(n_topics):
    print '\nTopic %s' % t
    for i in np.argsort(-doc_topic_probs, axis=0)[:,t][:5]:
        print doc_topic_probs[i,t], texts[i]

print '\nWord Assignment to Topics:'
for t in range(n_topics):
    print '\nTopic %s' % t
    for i in np.argsort(-word_topic_probs, axis=0)[:,t][:5]:
        print word_topic_probs[i,t], dictionary.id2token[i]


# TODO: Gibbs sampler burn in? How should the above change?


