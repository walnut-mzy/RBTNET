import os

"""
bert.py
"""
bert_path="model/pretrain/bert/"
initializer_range=0.02
#这个参数在concat_bert里面无用
class_nums_bert=None
config="model/pretrain/bert/config.json"
"""
image.py
"""
#这个参数在concat_bert里面无用
class_num_img=2
num_image_embeds=4
pool_func="avg"
# image_path="model/pretrain/image/resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5" #填入权重文件或者路径
image_path="imagenet" #填入权重文件或者路径
"""
resnet50.py
"""
image_path_resnet50="model/pretrain/image/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
"""
Inceptionv3.py
"""
image_path_Iceptionv3="model/pretrain/image/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
"""
concat_bert.py
"""
imageclf="resnet152"  #这个参数用来选择imageclf使用哪个主体模型（resnet50,resnet152.Inceptionv3）
txtclf="bert"       #这个参数用来选择txtclf使用哪个主体模型(bert，FastText,birnn)
placeclf="lstm"  #这个参数用来选择txtclf使用哪个主体模型(Transformer,lstm,lstm_cnn)
class_num=2
hide_size=768
isbn=True
dropout_lr=0.2
"""
place.py
"""
dn1=512
dn2=256
dn3=128
"""
data.py
"""
path_data="data/barrierData.txt"
path_img="data/barrierImages"
img_w=224
img_h=224
tokenize_path="model/pretrain/bert"
max_size=64
is_val=False
BUFFER_SIZE=2000
batch=4
repeat=3
noise_mode="gasuss"                    #这里支持两种mode:gasuss,Impulse
txt_mask=0                #这里指的是被mask的概率
mean=0
var=0
prob=0.1
"""
train
"""
hide_size_concat_bert=512
save_path="save"
log_dir="logs"
epoch=80
is_gpu= True
is_vit=True
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
"""
train_only.py
"""
is_only=True
is_txt=True
is_img=False
assert is_txt!=is_img,"is_txt parameters and is_val parameters are different and are both true or false"

"""
VilT.py
"""
num_classes = 2
image_size = 224
num_heads = 12
patch_size = 16
embed_dim = 768
layer_length = 12

"""
inference.py
"""
sentense_max_size=32