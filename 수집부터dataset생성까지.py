from multiprocessing_news_crawling import crawler, get_price
import nltk, tf_idf, random
import pandas as pd
docs_content = []
docs_title = []
docs_date = []
docs_code = []

# 제한적인 목록 규제
companies = ["005930", "051910","096770", "003620", "005360", "068270"]
# crawling_start_date 부터 현재까지 모을것이다.
crawling_start_date = "2020.06.01"


for CP in companies:
    CD = crawling_start_date
    print(CP, CD)
    PDDATA = crawler(CP, CD )
    docs_content.extend(PDDATA['기사내용'])
    docs_title.extend(PDDATA['기사제목'])
    docs_date.extend(PDDATA['날짜'])
    docs_code.extend(PDDATA['코드'])
    print(len(docs_title))
    print("complete", CP)
'''
문장별로 무엇이 출력될지 궁금하다면 시도하길 바란다.
docs = docs_title
words_and_importent_in_docs = tf_idf.tf_idf(docs)

for i, I in enumerate(docs):
    if random.randrange(10) != 0: continue
    SL =sorted(words_and_importent_in_docs[i].items(), key = lambda X : (X[1], X[0]))[::-1]
    print(I)
    print(SL[:5])
    print("\n")
'''
# 거래 대금을 대강 때려 맞춘다.
VALUE = lambda GP: [(D,int(V)*int(PR)) for D, V, PR in zip(GP["Date"], GP['Volume'], GP['Close'])]
CODE_AND_VALUE = {I:VALUE(get_price(I)) for I in companies}
import bisect
@nltk.memoize
def up_or_down(code, date):
    """
    전거래일 대비 다음거래일에서 거래대금이 증가할 날짜 :+1
    전거래일 대비 다음거래일에서 거래대금이 유지할 날짜 :0 사실상 없을것으로 예측됨
    전거래일 대비 다음거래일에서 거래대금이 감소할 날짜 :-1
    """
    V2 = bisect.bisect_right(CODE_AND_VALUE[code], (date, 0) )
    if V2 == len(CODE_AND_VALUE[code]): return None
    V1 = CODE_AND_VALUE[code][V2-1]
    V2 = CODE_AND_VALUE[code][V2]
    print(V1, V2)
    return -1 if V1[1] > V2[1] else int(V1[1] < V2[1])

def data_to_dataset(datas, code_and_dates = None, labels = None, return_format = "APPLY_FOR_NLTK_CLASSIFY"):
    """
    datas별로 dates와 codes를 제공 하거나
    code_and_dates 는 list속의 (code, date) 형식
    label을 내놓아야 한다.
    
    APPLY_FOR_NLTK_CLASSIFY의 경우
    참고로, 너무 최근이어서 다음날의 주가를 알수없는경우 featureset 만으로 2번 반환리스트에 속한다.
    그 이외에서는 (featureset, label) 로 1번 반환리스트에 속한다.
    """
    assert( (code_and_dates is not None) | (labels is not None ))
    if not labels:
        assert(len(datas) == len(code_and_dates))
        labels = [ up_or_down(*I) for I in code_and_dates]
    
    words_and_importent_in_docs = tf_idf.tf_idf(datas)
    def build_feature():
        return [sorted(I.items(), key = lambda X : (X[1], X[0]))[::-1][:5] for I in words_and_importent_in_docs.values()]
    features = build_feature()
    if return_format == "APPLY_FOR_NLTK_CLASSIFY":
        """
        if you want use full feature
        but it is extremely low efficiency
        """
        return [({W:V for W, V in F},L) for F, L in zip(features, labels) if L is not None], \
        		[{W:V for W, V in F} for F, L in zip(features, labels) if L is None]
        """
        존재성만 다룰경우의 파라메터 반환
        """
        return [({W:1 for i,(W, V) in enumerate(F)},L) for F, L in zip(features, labels) if L is not None], \
        		[{W:1 for i,(W, V) in enumerate(F)} for F, L in zip(features, labels) if L is None]
    else: return features, labels
    
dataset, un_trainable_data = data_to_dataset(docs_title, list(zip(docs_code, docs_date)))
random.shuffle(dataset)
train_set, test_set = dataset[:int(len(dataset)*.9)],dataset[int(len(dataset)*.9):]

"""
현재 tf_idf에서 데이터 수집을 하지 않는다. 그에 따라 즉흥적으로 뉴스를 추가하지 못한다.
기존의 뉴스 셋을 포함하여 알아내야 idf치역이 흔들리지 않으므로, 기존의 데이터를 항상 포함해야 한다.
"""

"""
이부분에서 너무 파라메터가 많아서 오래걸린다.
이걸 tensorflow로 처리바람.

아래는 지금 당장에서는 작동가능한 예제
#classifier = nltk.classify.DecisionTreeClassifier.train(train_set)
classifier = nltk.classify.DecisionTreeClassifier.train(train_set[:40])
print(nltk.classify.accuracy(classifier, test_set))
"""
