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
import numpy

data = pd.read_csv(r'C:\Users\password\Desktop\cs372_group-master (1)\cs372_group-master\dataset.csv', low_memory=False)
train_data = data.sample(frac=0.8, random_state = 2020)
test_data = data.drop(train_data.index)

print(train_data[:5])
print(test_data[:5])
print('총 train 샘플의 수 :',len(train_data))
print('총 test 샘플의 수 :',len(test_data))

train_data.drop_duplicates(subset = ['title'], inplace=True) # words 열에서 중복인 내용이 있다면 중복 제거
train_data['title'] = train_data['title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
train_data['title'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
train_data = train_data.dropna(how='any') # Null 값 제거
print(train_data[:5])
print('전처리 후 훈련용 샘플의 개수 :',len(train_data))

test_data.drop_duplicates(subset = ['title'], inplace=True) # words 열에서 중복인 내용이 있다면 중복 제거
test_data['title'] = test_data['title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['title'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print(test_data[:5])
print('전처리 후 훈련용 샘플의 개수 :',len(test_data))

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()
X_train = []
i = 1
for sentence in train_data['title']:
    i += 1
    if i % 100 == 0:
        print(i)
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)
        
X_test = []
j = 0
for sentence in test_data['title']:
    j += 1
    if j % 100 == 0:
        print(j)
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_test.append(temp_X)

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
X_test = tokenizer.texts_to_sequences(X_test)
print(X_train[:3])

y_train = np.array(train_data['label'] + 1)
y_test = np.array(test_data['label'] + 1)

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
drop_test = [index for index, sentence in enumerate(X_test) if len(sentence) < 1]
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
  
max_len = 18
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)
###############################################################################################
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import *

print(X_train[:5])
print(y_train[:5])
print("vocab_size = ", vocab_size)

IN_label = Input((max_len,), dtype = tf.uint32)
EMB = Embedding(vocab_size, max_len)
EMBED = EMB(tf.expand_dims(IN_label, -1))
EMBED = tf.squeeze(EMBED, -2)
#CONC = tf.concat([EMBED, tf.expand_dims(IN_value, -1)], -1)
CONC = EMBED

X = BatchNormalization()(Dense(10)(CONC))
X = ReLU()(X)

X = tf.transpose(X,[0,2,1])
print(X)
X = Bidirectional(LSTM(10, return_sequences = True))(X)
X = LayerNormalization()(X)
print(X)
A = Bidirectional(LSTM(20, return_sequences = True))(X)
X = Bidirectional(LSTM(10, return_sequences = True))(A) + X
X = LayerNormalization()(X)
print(X)
A = Bidirectional(LSTM(20, return_sequences = True))(X)
X = Bidirectional(LSTM(10, return_sequences = True))(A) + X
X = LayerNormalization()(X)
print(X)
X = LSTM(10)(X)
##X = LSTM(10)(tf.transpose(X,[0,2,1]))
#X = tf.concat(tf.unstack(X, axis = -1),-1)
#X = BatchNormalization()(Dense(10)(X))
#X = ReLU()(X)
X = Dense(3, 'softmax')(X)

model = Model(IN_label,X)
model.summary()
model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
##########################################################################################
#학습시키기
history = model.fit(X_train, y_train, epochs=15, batch_size=256, validation_split=0.2)

##########################################################################################
"""
# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file : 
    json_file.write(model_json)
    
# Save model weights
model.save_weights("best_model.h5")
print("Saved model to disk")
##########################################################################################

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
"""
# 그래프 그리기
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = numpy.arange(len(y_loss))
plt.ylim((0.0,1.5))
plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
plt.plot(x_len, y_loss, marker='.', c='yellow', label="Train-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

y_vacc = history.history['val_accuracy']
y_acc = history.history['accuracy']

x_len = numpy.arange(len(y_acc))
plt.ylim((0.5,1.0))
plt.plot(x_len, y_vacc, marker='.', c='green', label="Validation-set Acc")
plt.plot(x_len, y_acc, marker='.', c='blue', label="Train-set Acc")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


# 모델 평가
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))


sentence = "셀트리온, 1분기 영업이익 1202억원…전년比"
result = []
temp_X = []
temp_X = okt.morphs(sentence, stem=True)
temp_X = [word for word in temp_X if not word in stopwords]
result.append(temp_X)
result = tokenizer.texts_to_sequences(result)
result = pad_sequences(result, maxlen = max_len)


for score in model.predict(result):
    score = tf.argmax(score)
    if score == 2:
        print("긍정적입니다.")
    elif score == 0:
        print("부정적입니다.")
    else:
        print("중립적입니다.")
