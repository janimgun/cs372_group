from sklearn.manifold import TSNE
import sklearn
import nltk.corpus
from nltk.corpus import wordnet
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import umap.umap_ as umap
import random,time
random.seed(1339)#Covid19
epochs = 10

from word_filter import usable_and_filtered_words as UFW
raw_words = nltk.corpus.gutenberg.words(nltk.corpus.gutenberg.fileids()[0])
usable,lemma_dict, texts = UFW([I.lower() for I in raw_words])
from embedder import labeling


def random_pair_generator(*args,limit = -1):
    total = np.prod(args)
    while limit:
        limit -=1
        R = random.randrange(total)
        ret = []
        for I in args:
            ret.append(R%I)
            R//=I
        yield tuple(ret)
from nltk.stem import PorterStemmer
def cutter(X):
    # stem 이 더 많이 줄임
    return PorterStemmer().stem(X)
def print_groups(domain,words,labels,mute = True,similarity_MAT = None):
    def avr_dist(X):
        test_pairs = set(random_pair_generator(len(X),len(X),limit = 100))
        return np.mean([dist(words[I[0]],words[I[1]]) for I in test_pairs if I[0]<I[1] ] )
    group_idx = {} # stem-label
    for W,L in zip(texts, labels):
        if L==-1: continue
        group_idx[cutter(W)] = L
    marged_domains = {} # label-단어들
    print('except group -1::')
    for I in domain:
        idx = group_idx.get(cutter(I),-1)
        if idx == -1: continue #except group -1
        if marged_domains.get(idx,None):marged_domains[idx].append(I)
        else: marged_domains[idx] = [I]
    marged_idx = {} # label-idx
    for i,I in enumerate(words):
        idx = group_idx.get(cutter(I),-1)
        if idx == -1: continue #except group -1
        if marged_idx.get(idx,None):marged_idx[idx].append(i)
        else: marged_idx[idx] = [i]
    
    similarity_AVR = None
    if similarity_MAT is not None:
        similarity_AVR = [0 for I in range(len(marged_idx))]
        for K,V in marged_idx.items():
            total = 0.0
            cnt = 0
            for I in V:
                cnt += (similarity_MAT[I,V] != -1).sum()
                total += np.where(similarity_MAT[I,V] > -1\
                                     , similarity_MAT[I,V],0).sum()
            similarity_AVR[K] = total / max(1,cnt)
    AVRDIST = [avr_dist(V) for K,V in sorted(marged_domains.items())]
    if mute == False:
        #for K,V in sorted(marged_domains.items()):print("{} size = {}".format(K,len(V)))
        for i,[K,V] in enumerate(sorted(marged_domains.items())):
            AVRD = AVRDIST[i]
            if similarity_MAT is not None:
                SIMIL = similarity_AVR[i]
                print("{:<5}::size ={:<4} avrdist = {:<5} similar = {:<5}\n{}\n".format(K,len(V),AVRD,SIMIL,str(V)))
            else:
                print("{:<5}::size ={:<4} {:<5}\n{}\n".format(K,len(V),AVRD,str(V)))
    else:
        GROUP_sizes = [len(V) for K,V in marged_domains.items()]
        print("{:<10} avr {:<10} std {:<10}".format("size",np.mean(GROUP_sizes),np.std(GROUP_sizes)))
        print("{:<10} avr {:<10} std {:<10}".format("dist",np.mean(AVRDIST),np.std(AVRDIST)))
        if similarity_MAT is not None:
            print("{:<10} avr {:<10} std {:<10}".format("similarity", np.mean(similarity_AVR),np.std(similarity_AVR)))
def statistics_for_word(W):
    # input as string
    print("{:<20} in texts".format('is' if W in texts else 'is not'))
    print("{:<20} in usable".format('is' if W in usable else 'is not'))
    order = sum(i if cutter(I) == cutter(W) else -1 for i,I in enumerate(texts)) + len(texts) -1
    if order < 0:return None
    print('order is {:<20} if exist as "{}"'.format(order,texts[order]) )
    label = clusterer.labels_[order]
    print('label is {:<20}'.format(label))
    s_mat = distance_value.similarity[order]
    calced = sum(s_mat != -1)
    print("{:<5}({:>5}%) calculated and {:<5} is related".format(\
                        calced, calced*10000//len(texts)/100,sum(s_mat > 0)) )
    LI = [I for I in s_mat if I != -1]
    print('avr = {:<20} std = {:<20}'.format(np.mean(LI),np.std(LI)))
    return order
def group_stats(group):
    orders = [statistics_for_word(I) for I in group]
    for I in orders:
        print(distance_value.similarity[I,orders])
def print_group_stats(labels):
    group_size = dict(nltk.FreqDist(labels))
    group_size[-1] = group_size.get(-1,0)
    X = np.array(list(V for K,V in group_size.items() if K != -1) or [0])
    print("cnt = {:<5}  -1 = {:<5} :: max = {:<5}, mean = {:<5.3f}, min = {:<5}, std = {:<5.3f}".format(\
                X.shape[0],group_size[-1],X.max(), X.mean(), X.min(), X.std()))
try: 
    labeling(texts,1,1,function = print_group_stats, reset = False, bug_control_mod = True)
    # if no problem
    labels = labeling(texts,epochs,30,function = print_group_stats, reset = True, bug_control_mod = False)
except: 
    print("try to control error")
    labels = labeling(texts,epochs,30,function = print_group_stats, reset = True, bug_control_mod = False)
    #labels = labeling(texts,epochs,30,function = print_group_stats, reset = False, bug_control_mod = False)

#print_groups(domain=texts,words=texts\
#            ,labels=labels,mute=False,similarity_MAT = distance_value.similarity)
#D = dict(nltk.FreqDist(clusterer.labels_))
#print(D)
