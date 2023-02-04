import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv2D,Input
from tensorflow.keras import Sequential, Model
import setting
class place_dn_encoder(tf.keras.layers.Layer):
    def __init__(self,hidesize):
        super(place_dn_encoder, self).__init__()
        self.dn1 = tf.keras.layers.Dense(hidesize)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.Dp = tf.keras.layers.Dropout(0.2)

    def call(self, inputs):
        x=self.dn1(inputs)
        x=self.bn(x)
        x=self.relu(x)
        x=self.Dp(x)
        return x
class place_dn(tf.keras.layers.Layer):
    def __init__(self):
        super(place_dn, self).__init__()
        self.place_encoder=place_dn_encoder(setting.dn1)
        self.place_encoder2=place_dn_encoder(setting.dn2)
        self.place_encoder3=place_dn_encoder(setting.dn3)
    def call(self, inputs):
        x=self.place_encoder(inputs)
        x=self.place_encoder2(x)
        x=self.place_encoder3(x)
        return x