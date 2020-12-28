from PIL import Image
from os import listdir
from os import walk
from os.path import isfile, isdir, join
from keras.datasets import mnist
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

#預測歷史
class LossHistory(keras.callbacks.Callback):
    def __init__(self,model,x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.i=0
    """
    def on_epoch_end(self, epoch, logs=None):
        self.i+=1
        print("\ntime:"+str(i))
    """
    def on_train_end(self, logs=None):
        y_test_pred = np.argmax(self.model.predict(self.x_test), axis=1)
        error=0
        for i in range(len(y_test)):
            if(y_test_pred[i]!=y_test[i]):
                error+=1
        print("\n106502027's Test Accurance: ",(len(y_test)*1.0-(error*1.0))/len(y_test))

#定義圖片大小
img_rows, img_cols = 28, 28
#讀取訓練資料路徑
train_path = "./image"
files = listdir(train_path)
x_data=np.array([])
data_number=0
#讀取資料夾內所有檔案
for root, dirs, files in walk(train_path):
    for f in files:
        if data_number==0:
            fullpath = join(root, f)
            im = Image.open(fullpath)
            x_data = (np.array(im) / 255).reshape(1,28,28)  # 讀取資料時順便做資料正規化
            #print(fullpath)
            #print(x_data.shape)
            data_number += 1
        else:
            fullpath = join(root, f)
            im = Image.open(fullpath)
            im = (np.array(im)/255).reshape(1,28,28)
            x_data = np.vstack((x_data,im)) # 讀取資料時順便做資料正規化
            #print(fullpath)
            #print(x_data.shape)
            data_number += 1
x_data=x_data.reshape(data_number,img_rows,img_cols,1) #調整資料格式
#建立label
y_data=[]
for k in range(0,49,1):
    for i in range(0,10,1):
        for j in range(0,5,1):
            y_data.append(i)

#讀取測試資料
test_path = "./image"
files = listdir(test_path)
x_test=[]
test_number=0
for root, dirs, files in walk(test_path):
    for f in files:
        fullpath = join(root, f)
        im = Image.open(fullpath)
        p = np.array(im)/255
        x_test.append(p)
        test_number+=1
x_test=np.array(x_test)
x_test=x_test.reshape(test_number,img_rows,img_cols,1)

#建立test_label
y_test=[]
for k in range(0,34,1):
    for i in range(0,10,1):
        for j in range(0,5,1):
            y_test.append(i)
#資料參數
num_classes = 10
epochs = 3
input_shape=(img_rows,img_cols,1)
#one hot encoding
y_train = keras.utils.to_categorical(y_data, num_classes)

#建立模型
model = Sequential()
model.add(Conv2D(2, kernel_size=(1, 1),
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
#設定model參數
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
#設定訓練紀錄
history = LossHistory(model, x_test, y_test)
#開始訓練
model.fit(x_data, y_train,
          epochs=epochs,
          verbose=2,
          shuffle=True,
          callbacks=[history])

pre=model.predict(x_data)
print(np.argmax(pre, axis=1))
print('test before save: ', model.predict(x_test[0:2]))
model.save('my_model.h5')
del model

model = load_model('my_model.h5')
print('test after load: ', model.predict(x_test[0:2]))