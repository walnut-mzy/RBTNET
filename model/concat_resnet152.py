import tensorflow as tf
from model.bert import Bertclf
import setting
from model.image import imageclf
if setting.txtclf=="FastText":
    from model.fast_text import FastText
elif setting.txtclf=="birnn":
    from model.birnn import TextBiRNN
elif setting.txtclf=="rcnn":
    from model.rcnn import RCNN
else:
    assert "库内没有该模型"+str(setting.txtclf)
from tensorflow.keras.layers import Layer,Conv2D,Input
from tensorflow.keras import Sequential, Model
from model.palce import place_dn
class_num=2
hide_size=768
isbn=True
class MultimodalConcatBertClf(tf.keras.layers.Layer):
    def __init__(self,len_dataset,vocab_size,hide_size,class_num,isbn=True):
        super(MultimodalConcatBertClf, self).__init__()

        self.txtEncoder=None
       # self.BertEncoder=Bertclf(bert_path=setting.bert_path,isencoder=True,initializer_range=setting.initializer_range,class_nums=setting.class_nums_bert)
        if setting.txtclf=="FastText":
            self.txtEncoder=FastText(
                maxlen=len_dataset,
                max_features= vocab_size,
                embedding_dims=512,
            )
        elif setting.txtclf=="birnn":
            self.txtEncoder=TextBiRNN(
                maxlen=len_dataset,
                max_features=vocab_size,
                embedding_dims=512,
            )
        elif setting.txtclf=="rcnn":
            self.txtEncoder=RCNN(
                maxlen=len_dataset,
                max_features=vocab_size,
                embedding_dims=512,
            )
        self.imageEncoder=imageclf(pool_func=setting.pool_func,class_num=setting.class_num_img,num_image_embeds=setting.num_image_embeds,isencoder=True )
        self.PlaceEncoder=place_dn()
        self.dn1=tf.keras.layers.Dense(hide_size)
        self.dn2=tf.keras.layers.Dense(class_num,activation="softmax")
        self.bn=tf.keras.layers.BatchNormalization()
        self.relu=tf.keras.layers.ReLU()
        self.fl=tf.keras.layers.Flatten()
        self.Dp=tf.keras.layers.Dropout(setting.dropout_lr)
        self.isbn=isbn
    def call(self,inputs):

        txt_content=inputs["txt"]
        img_content=inputs["img"]
        place_content=inputs["place"]


        img=self.imageEncoder(img_content)
        img=self.fl(img)
        place=self.PlaceEncoder(place_content)
        txt=self.txtEncoder(txt_content)
        out=tf.concat([txt,img,place],axis=-1)
        x=self.dn1(out)
        if self.isbn:
            x=self.bn(x)
        x=self.relu(x)
        x=self.Dp(x)
        x=self.dn2(x)
        return x
# if __name__ == '__main__':
#     inputs = Input(shape=(224,224,3), batch_size=2)
#     inputs2= tf.keras.layers.Input(shape=(64),dtype=tf.int32,batch_size=2)
#     inputs3 = tf.keras.layers.Input(shape=(64), dtype=tf.int32, batch_size=2)
#     inputs4 = tf.keras.layers.Input(shape=(64), dtype=tf.int32, batch_size=2)
#     inputs5 = tf.keras.layers.Input(shape=(2), dtype=tf.int32, batch_size=2)
#     imageclf=MultimodalConcatBertClf(hide_size=hide_size,class_num=class_num,isbn=isbn,vocab_size=21128)
#     input={
#         "place":inputs5,
#         "img":inputs,
#         "txt":inputs2
#     }
#     out = imageclf(input)
#     model = Model(inputs=input, outputs=out, name='concat_bert-tf2')
#     model.summary()