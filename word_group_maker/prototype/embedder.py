import umap.umap_ as umap
import hdbscan
import scipy
import numpy as np
import random,time,math
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
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
class adj_list:
    def __init__(self,word_list, base_setting = 50 * 2):
        self.word_list = word_list
        self.similarity = [{i:1.0} for i in range(len(word_list))]
        self.sampled_similarity = np.ndarray((len(word_list),))*0
        self.sampled_count = np.ndarray((len(word_list),),dtype = np.int64)*0
        self.global_random(lowerbound_to_assign = base_setting,recording = True)
    def distance(self):
        #factor_matrix = scipy.sparse.lil_matrix(\
        #            len(self.similarity), max(len(I) for I in self.similarity), dtype=np.float32)
        factor_matrix = scipy.sparse.lil_matrix(\
                    (len(self.similarity), len(self.similarity)), dtype=np.float32)
        factor_matrix.rows = np.array([list(I.keys()) for I in self.similarity])
        # 이건 feature 값이고, 계산 안된곳에는 mean을 배치하기위해 상대값을 조정하여 avr 을 배치않고 전체를 avr로 뺌
        average_each_row = self.sampled_similarity / self.sampled_count
        factor_matrix.data = np.array([[J - AVR for J in I.values()] for I,AVR in zip(self.similarity,average_each_row)])
        return factor_matrix
    def grouped_update(self, index):
        grouped = [[]for _ in range(max(index)+1)]
        for i,I in enumerate(index):
            if I !=-1: grouped[I].append(i)
        total_calculation = 0
        for I in grouped:total_calculation += self.similarity_update(I)
        print('calculate {:<6} added'.format(total_calculation))
        return total_calculation
    def global_random(self,random_process = None, lowerbound_to_assign = 0,recording = False):
        if recording == True:
            bf_cnt = np.array([len(I) for I in self.similarity])
            bf_sim = np.array([sum(I.values()) for I in self.similarity])
        total_calculation = self.similarity_update(range(len(self.word_list))\
                    ,random_process= random_process, lowerbound_to_assign = lowerbound_to_assign)
        if recording == True:
            self.sampled_count += np.array([len(I) for I in self.similarity]) - bf_cnt
            self.sampled_similarity += np.array([sum(I.values()) for I in self.similarity]) - bf_sim
        
        print('calculate {:<6} added'.format(total_calculation))
        return total_calculation
    def similar_value(self,idx1,idx2):
        def _sub(S1,S2):
            return max(S1.wup_similarity(S2) or 0.0,S2.wup_similarity(S1) or 0.0)
        similar = [_sub(I,J) for I in wordnet.synsets(self.word_list[idx1])[:2]
            for J in wordnet.synsets(self.word_list[idx2])[:2]]
        if len(similar) == 0: return 0.0
        return np.mean(sorted(similar)[-2:])
    def similarity_update(self, group_index, random_process = None, lowerbound_to_assign = 0):
        if len(group_index) <= 50:
            test_pairs = [(i,j) for i in range(len(group_index)) for j in range(i+1,len(group_index))]
        else:
            if random_process == None:random_process = int( len(group_index) **.5 * 250 ) 
            test_pairs = set(random_pair_generator(len(group_index),len(group_index)\
                                                 ,limit = random_process))
        total_calculation = 0
        for [i,j] in test_pairs:
            I,J = group_index[i],group_index[j]
            if self.similarity[I].get(J,-1)==-1:
                self.similarity[J][I] = self.similarity[I][J] = self.similar_value(I,J)
                total_calculation += 1
        if lowerbound_to_assign:
            print("start to assign for lower term ")
            for I in group_index:
                RPG = random_pair_generator(len(group_index)\
                    , limit = max(0,lowerbound_to_assign - len(self.similarity[I])) )
                for j in RPG:
                    J = group_index[j[0]]
                    if self.similarity[I].get(J,-1)==-1:
                        self.similarity[J][I] = self.similarity[I][J] = self.similar_value(I,J)
                        total_calculation += 1
        return total_calculation
    def similarity_update_except_group(self, index, random_process = None):
        if random_process == None:random_process = int( len(self.word_list) **.5 * 250 ) 
        test_pairs = set(random_pair_generator(len(self.word_list),len(self.word_list)\
                                             ,limit = random_process))
        total_calculation = 0
        for [I,J] in test_pairs:
            if index[I] != index[J] and self.similarity[I].get(J,-1)==-1:
                self.similarity[J][I] = self.similarity[I][J] = self.similar_value(I,J)
                total_calculation += 1
        print('calculate {:<6} added'.format(total_calculation))
        return total_calculation
def labeling(words,epochs,minimum_groups = 10,function = None, reset = True, bug_control_mod = False,metric = 'cosine' ):
    # canberra avr = .20 slow 117s/epoch
    # manhattan avr = .36, cnt = 2 50s/epo 
    # braycurtis,haversine,seuclidean not sparse
    # correlation cnt = 13 .36 slow 110s / epoch
    # euclidean = .3
    # cosine  avr = 0.46961 std = 0.12803 37s/epo
    
    def plotting(XX,YY,reguard_half = 20):
        def move(X):
            x = np.array(X)
            return x -(x.min()+x.max())/2
        X = move(XX)
        Y = move(YY)
        plt.grid(True,color = 'k')
        plt.scatter(X,Y, 9,'k')
        plt.scatter(X,Y, s = 1,cmap = 'Spectral', c = labels)
        half = max(reguard_half,X.max(),Y.max())
        plt.xlim(-half,half)
        plt.ylim(-half,half)
        plt.show()
    def sim_each_group(labels):
        cluster_idx ={}
        for i,I in enumerate(labels):
            if cluster_idx.get(I,-1)==-1: cluster_idx[I]=[i]
            else: cluster_idx[I].append(i)
        sim_value = {}
        tot_sim = 0.0
        for [K,Vi] in sorted(cluster_idx.items()):
            sim_sco =0.0
            cnt = 0
            for I1 in Vi:
                for I2 in Vi:
                    if I1 != I2:
                        sim=distance_value.similarity[I1].get(I2,-1)
                        if sim>=0:sim_sco+=sim;cnt+=1
            sim_sco/=max(1,cnt)
            sim_value[K] = sim_sco
        return sim_value
    begin_time = time.time()
    DV = globals().get('distance_value', None)
    print(DV)
    global distance_value
    distance_value = adj_list(words, **({'base_setting' : 2} if bug_control_mod else {}))\
                                            if reset or DV is None else DV 
    
    epo = 0
    stables = 0
    
    X_embedded = 'spectral'
    while True:
        X_trans = distance_value.distance()
        print('epoch{:<3}, {:<5.3f} spend, goto umap'.format(epo, time.time()-begin_time))
        #model = umap.UMAP(metric = 'cosine', n_epochs = 800, unique = True)#, init = X_embedded)
        model = umap.UMAP(metric = metric, n_neighbors = 20, init = X_embedded).fit(X_trans)
        
        X_embedded = model.embedding_
        clusterer = hdbscan.HDBSCAN()
        clusterer.fit(X_embedded)
        
        labels = clusterer.labels_
        #umap.plot.points(model, labels = labels)
        plotting(X_embedded[:,0],X_embedded[:,1])
        if function:
            function(labels=labels)
        
        epo += 1
        
        #total_calculation = distance_value.grouped_update(labels)
        '''
        total_calculation = distance_value.similarity_update_except_group(labels)
        if total_calculation<1000: distance_value.global_random(10000)
        if epo>=epochs/2:total_calculation = distance_value.grouped_update(labels)
        '''
        distance_value.global_random(10000)
        #if epo%5==0 or epo >= epochs-2 :total_calculation = distance_value.grouped_update(labels)
        total_calculation = distance_value.grouped_update(labels)
        
        sim_scores = sim_each_group(labels)
        sim_stat = np.array([V for K,V in sim_scores.items() if K!=-1])
        print("avr = {:2.5f} std = {:2.5f}\n".format(sim_stat.mean(),sim_stat.std()))
        if max(labels) + 1 >= minimum_groups:
            stables+=1
            if epo >= epochs:
                if stables >= 2:
                    cluster_word ={}
                    for i,[I,J] in enumerate(zip(labels,distance_value.word_list)):
                        if cluster_word.get(I,-1)==-1: cluster_word[I]=[J]
                        else: cluster_word[I].append(J)
                    tot_sim = 0.0
                    for [Kss,sim_sco],[K,Vw] in zip(sorted(sim_scores.items()),sorted(cluster_word.items())):
                        tot_sim += sim_sco
                        print("{:<4} :: size = {:<4} sim = {:1.4f}\n{}\n".format(K,len(Vw),sim_sco,str(Vw)))
                    print("avr sim = {}".format(tot_sim/len(cluster_word)))
                    return labels
