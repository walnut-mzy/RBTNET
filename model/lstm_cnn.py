
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Layer,Conv1D,Input,LSTM,Dense,Embedding,GlobalAveragePooling1D,Dropout,MaxPooling1D
from tensorflow.keras import Sequential, Model

import setting
class lstm_cnn(tf.keras.layers.Layer):
    def __init__(self):
        super(lstm_cnn, self).__init__()
        self.lstm1=LSTM(128,return_sequences=True)
        self.emb=Embedding(input_dim=2, output_dim=128)
        self.cn=Conv1D(256,kernel_size=1,padding="same")
        self.max=MaxPooling1D()
        self.fn1=Dense(256,activation="relu")
        self.dp1=Dropout(0.3)
    def call(self,inputs):
        x=self.emb(inputs)
        x=self.lstm1(x)
        x=self.cn(x)
        x=self.max(x)
        x=tf.squeeze(x,axis=1)
        x=self.fn1(x)
        x=self.dp1(x)
        return x
if __name__ == '__main__':
    inputs = Input(shape=(2), batch_size=16)
    imageclf=lstm_cnn()
    out = imageclf(inputs)
    model = Model(inputs=inputs, outputs=out, name='SRCNN-tf2')
    model.summary()