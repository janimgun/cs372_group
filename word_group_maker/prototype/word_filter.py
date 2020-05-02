
def usable_and_filtered_words(raw_words):
    import nltk 
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk.stem import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    def cutter(X):
        thing = nltk.FreqDist([I.pos() for I in wordnet.synsets(X)][:5]).max()
        return WordNetLemmatizer().lemmatize(X, pos=thing)
    stopwords=[]#set(nltk.corpus.stopwords.words('english'))
    def filtering(X):
        if X in stopwords : return None
        if len(wordnet.synsets(X)) <= 1 : return None
        if not X.isalpha(): return None
        if len(X) <= 2: return None
        return X
    def filtering_words(word_list):
        def use_in_list(lem,word_list):
            if len(wordnet.synsets(lem)) : return lem
            if len(word_list)==1:return word_list[0]
            return sorted((len(wordnet.synsets(I)),I) for I in word_list)[-1][1]
        print("total texts taken = {}".format(len(word_list)))
        words = set(word_list)
        print("total non-duplicated words = {}".format(len(words)))
        words = list(filter(filtering,words))
        print("filtered words = {}".format(len(words)))
        usable = words 
        lemma_dict = {}
        for I in words:
            lem = cutter(I)
            if lemma_dict.get(lem,None):
                lemma_dict[lem].append(I)
            else: lemma_dict[lem] = [I]
        words = list(use_in_list(K,V) for K,V in lemma_dict.items())
        print("cutting non-duplicated words = {}".format(len(lemma_dict)))
        return usable,lemma_dict, words
    return filtering_words(raw_words)
