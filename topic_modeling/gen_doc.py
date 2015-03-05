import numpy as np
from numpy.random import multinomial

def sigmoid(x):
    return 1./(1+np.exp(-x))

num_docs = 50

# Generate sample docs:
#topic_probs = [i/16. for i in xrange(17)]
topic_probs = [sigmoid(x) for x in [i/5. for i in xrange(-25, 25)]]
topic_terms = [['money', 'loan', 'bank'], ['river', 'stream', 'bank']]
texts = []

for doc in xrange(50):
    doc_terms = []
    for w in xrange(16):
        topic = int(np.where(multinomial(1, [topic_probs[doc], 1-topic_probs[doc]], size=1)[0] > 0)[0])
        word = int(np.where(multinomial(1, [1./3]*3, size=1)[0] > 0)[0])
        doc_terms.append(topic_terms[topic][word])
    texts.append(' '.join(doc_terms))

print texts



