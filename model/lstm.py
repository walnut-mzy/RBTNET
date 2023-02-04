import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Layer,Conv2D,Input,LSTM,Dense,Embedding,GlobalAveragePooling1D,Dropout
from tensorflow.keras import Sequential, Model

import setting

class lstm(tf.keras.layers.Layer):
    def __init__(self):
        super(lstm, self).__init__()
        self.token_emb = Embedding(input_dim=2, output_dim=128)
        self.lstm1=LSTM(128,return_sequences=True)
        self.lstm2=LSTM(512,return_sequences=True)
        self.lstm3=LSTM(512,return_sequences=True)
        self.lstm4=LSTM(512,return_sequences=True)
        self.lstm5=LSTM(128)
        self.fn1=Dense(256,activation="relu")
        self.fn2=Dense(128,activation="relu")
        self.fn3=Dense(64,activation="relu")
        self.dp1=Dropout(0.3)
        self.dp2 = Dropout(0.3)
        self.dp3 = Dropout(0.3)

    def call(self,inputs):
        x=self.token_emb(inputs)
        x=self.lstm1(x)
        x=self.lstm2(x)
        x1=self.lstm3(x)
        x=tf.concat([x, x1], axis=-1)
        x1=self.lstm4(x)
        x=tf.concat([x,x1],axis=-1)
        x=self.lstm5(x)
        x=self.fn1(x)
        x=self.dp1(x)
        x=self.fn2(x)
        x=self.dp2(x)
        x=self.fn3(x)
        x=self.dp3(x)

        return x
# if __name__ == '__main__':
#     inputs = Input(shape=(2), batch_size=16)
#     imageclf=lstm()
#     out = imageclf(inputs)
#     model = Model(inputs=inputs, outputs=out, name='SRCNN-tf2')
#     model.summary()