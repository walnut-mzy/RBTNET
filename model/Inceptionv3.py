import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Layer,Conv2D,Input
from tensorflow.keras import Sequential, Model
import setting
class_num=2
num_image_embeds=4
pool_func="avg"
image_path="pretrain/image/resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5" #填入权重文件或者路径
class imageclf(tf.keras.layers.Layer):
    def __init__(self,class_num,num_image_embeds,pool_func="avg",isencoder=False):
        super(imageclf, self).__init__()
        self.pool_func=None
        self.image_path=setting.image_path_Iceptionv3
        self.isencoder=isencoder
        if pool_func=="avg":
            self.pool_func=tf.keras.layers.AveragePooling2D
        elif pool_func=="max":
            self.pool_func = tf.keras.layers.MaxPooling2D
        assert self.pool_func!= None, "There is no such thing as an average pooling option"
        if num_image_embeds in [1,2,3,5,7]:
            self.pool_func(pool_size=[num_image_embeds,2])
        elif num_image_embeds==4:
            self.pool_func(pool_size=[2,2])
        elif num_image_embeds==6:
            self.pool_func(pool_size=[3, 2])
        elif num_image_embeds==8:
            self.pool_func(pool_size=[4, 2])
        elif num_image_embeds == 9:
            self.pool_func(pool_size=[3, 3])
        assert num_image_embeds<=9, "Illegal num_image_embeds"
        if self.isencoder==False:
            self.InceptionV3 = InceptionV3(classes=class_num, pooling=self.pool_func, classifier_activation='softmax',
                                        weights=self.image_path)
        else:
            self.InceptionV3 = InceptionV3(classes=class_num, pooling=self.pool_func, classifier_activation='relu',
                                        weights=self.image_path, include_top=False)
    def call(self, inputs):
        return self.InceptionV3(inputs)