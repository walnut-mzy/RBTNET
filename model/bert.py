from transformers import TFBertModel,BertConfig
import tensorflow as tf
import setting
bert_path="pretrain/bert/"
bert_config="pretrain/bert/config.json"
initializer_range=0.02
class_nums=2
class Bertclf(tf.keras.layers.Layer):
    def __init__(self,bert_path,initializer_range,class_nums,isencoder=False):
        super(Bertclf, self).__init__()
        self.class_nums=class_nums
        self.isencoder=isencoder
        self.initializer_range=initializer_range
        self.initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range)
       # config = BertConfig.from_json_file(bert_config)
        self.bert=TFBertModel.from_pretrained(bert_path)
        self.dn1=tf.keras.layers.Dense(self.class_nums,bias_initializer=self.initializer,activation="softmax")
    def call(self, inputs):
        # inputs=inputs.shape
        batch_x = inputs["input_ids"]
        batch_mask = inputs["token_type_ids"]
        batch_segment = inputs["attention_mask"]
        x=self.bert(batch_x,batch_mask,batch_segment)
        x=x["pooler_output"]
        if self.isencoder==False:
            return self.dn1(x)
        else:
            return x
if __name__ == '__main__':
    inputs1 = tf.keras.layers.Input(shape=(64),dtype=tf.float32)
    inputs2=tf.keras.layers.Input(shape=(64), dtype=tf.float32)
    inputs3=tf.keras.layers.Input(shape=(64),dtype=tf.float32)
    bert=Bertclf(bert_path,initializer_range,class_nums)
    out = bert({"input_ids":inputs1,
                 "token_type_ids":inputs2,
                 "attention_mask":inputs3
                 })
    inputs={"input_ids":inputs1,
                 "token_type_ids":inputs2,
                 "attention_mask":inputs3
                 }
    model = tf.keras.Model(inputs=inputs, outputs=out, name='SRCNN-tf2')
    model.summary()