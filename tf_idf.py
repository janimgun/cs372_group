import nltk, math
from konlpy.tag import *
tagger = Okt()
def tf_idf(docs):
    def tf():
        def words(doc): return tagger.pos(doc)
        return {i:dict(nltk.FreqDist(words(I))) for i, I in enumerate(docs)}
    #TF[문장번호][단어] = 로그빈도
    TF = {i:{j:math.log(J+1) for j,J in I.items()} for i,I in tf().items()} 
    
    TOTAL = 0
    freq = nltk.defaultdict(lambda : 0)
    for i,I in TF.items(): TOTAL += sum(I.values())
    for i,I in TF.items():
        for j,J in I.items():
            freq[j] += J
    #idf[단어] = 역빈도
    idf = {}
    for I,J in freq.items():
        idf[I] = math.log(TOTAL / freq[I])
    
    return {i:{j:J * idf[j] for j,J in I.items()} for i,I in TF.items()} 
