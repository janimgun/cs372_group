import pandas as pd
import json
import urllib.request
%matplotlib inline
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_data = pd.read_csv(r'C:\Users\password\Desktop\cs372_group-master (1)\cs372_group-master\dataset.csv', low_memory=False)
words = pd.Series([])
for i in range(len(train_data)):
    words.loc[i] = train_data.iloc[i]['F1_name'] + ' ' + train_data.iloc[i]['F2_name'] + ' ' + train_data.iloc[i]['F3_name'] + ' ' + train_data.iloc[i]['F4_name'] + ' ' + train_data.iloc[i]['F5_name']
train_data.loc[:, 'words'] = words

print(train_data[:5])
print('총 샘플의 수 :',len(train_data))

#train_data.drop_duplicates(subset = ['words'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
train_data['words'] = train_data['words'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
train_data['words'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
train_data = train_data.dropna(how='any') # Null 값 제거
print(train_data[:5])
print('전처리 후 훈련용 샘플의 개수 :',len(train_data))

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()
X_train = []
i = 1
for sentence in train_data['words']:
    i += 1
    if i % 100 == 0:
        print(i)
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)
    

###########################################################################################
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = total_cnt - rare_cnt + 1 # 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거. 0번 패딩 토큰을 고려하여 +1
print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
print(X_train[:3])

y_train = np.array(train_data['label'])

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
# 빈 샘플들을 제거
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))

print('단어셋의 최대 길이 :',max(len(l) for l in X_train))
print('단어셋의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))
  
max_len = 30
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen = max_len)
###############################################################################################
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='tanh'))

##########################################################################################
#학습시키기
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)
##########################################################################################
# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file : 
    json_file.write(model_json)
    
# Save model weights
model.save_weights("best_model.h5")
print("Saved model to disk")
##########################################################################################
"""
# model load
from keras.models import model_from_json 
json_file = open("model.json", "r")
loaded_model_json = json_file.read() 
json_file.close()
import tensorflow as tf
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
 
# model weight load 
loaded_model.load_weights("best_model.h5")
print("Loaded model from disk")

# 모델 평가
loaded_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
"""

sentence = "셀트리온, 계열사와 2000억규모 바이오시밀러 공급계약"
result = []
temp_X = []
temp_X = okt.morphs(sentence, stem=True)
temp_X = [word for word in temp_X if not word in stopwords]
result.append(temp_X)
result = tokenizer.texts_to_sequences(result)
result = pad_sequences(result, maxlen = max_len)


for score in model.predict(result):
    print(score)
    if score > 0.2:
        print("긍정적입니다.")
    elif score < -0.2:
        print("부정적입니다.")
    else:
        print("중립적입니다.")
