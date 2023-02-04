import tensorflow as tf
from model.bert import Bertclf
import setting
if setting.placeclf=="Transformer":
    from model.transformer import Transformer as placeclf
elif setting.placeclf=="lstm":
    from model.lstm import lstm as placeclf
elif setting.imageclf=="lstm_cnn":
    from model.lstm_cnn import lstm_cnn as placeclf
from model.lstm_cnn import lstm_cnn as placeclf
from model.image import imageclf
from tensorflow.keras.layers import Layer,Conv2D,Input
from tensorflow.keras import Sequential, Model
from model.palce import place_dn
from model.transformer import Transformer
class_num=2
hide_size=768
isbn=True

class MultimodalConcatBertClf(tf.keras.layers.Layer):
    def __init__(self,hide_size,class_num,isbn=True):
        super(MultimodalConcatBertClf, self).__init__()
        self.BertEncoder=Bertclf(bert_path=setting.bert_path,isencoder=True,initializer_range=setting.initializer_range,class_nums=class_num)
        self.imageEncoder=imageclf(pool_func=setting.pool_func,class_num=setting.class_num_img,num_image_embeds=setting.num_image_embeds,isencoder=True )
        self.PlaceEncoder=placeclf()
        self.dn1=tf.keras.layers.Dense(hide_size)
        self.dn2=tf.keras.layers.Dense(class_num,activation="softmax")
        self.bn=tf.keras.layers.BatchNormalization()
        self.relu=tf.keras.layers.ReLU()
        self.fl=tf.keras.layers.Flatten()
        self.Dp=tf.keras.layers.Dropout(setting.dropout_lr)
        self.isbn=isbn
    def call(self,inputs):

        input_ids=inputs["input_ids"]
        token_type_ids=inputs["token_type_ids"]
        attention_mask=inputs["attention_mask"]
        img_content=inputs["img"]
        place_content=inputs["place"]

        txt=self.BertEncoder({
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask
            })
        img=self.imageEncoder(img_content)
        img=self.fl(img)
        place=self.PlaceEncoder(place_content)

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
#     imageclf=MultimodalConcatBertClf(hide_size=hide_size,class_num=class_num,isbn=isbn)
#     input={
#         "place":inputs5,
#         "img":inputs,
#         "txt":{"input_ids":inputs2,
#                "token_type_ids":inputs3,
#                "attention_mask":inputs4,
#                  }
#     }
#     out = imageclf(input)
#     model = Model(inputs=input, outputs=out, name='concat_bert-tf2')
#     model.summary()