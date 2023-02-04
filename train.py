import os
import setting
import re
import sys
import GPUtil
from model.VilT import ViLTransformer
if setting.is_gpu==False:
    print("CPU已被使用")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from train_only import main
from tensorflow.keras.layers import Layer,Conv2D,Input
from tensorflow.keras import Sequential, Model
if setting.txtclf!="bert":
    from model.concat_resnet152 import MultimodalConcatBertClf
else:
    from model.concat_bert import MultimodalConcatBertClf
from data import Dataset
if setting.is_gpu:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    z = GPUtil.getGPUs()
    print("使用的gpu为:",z[0].name)
    print("gpu负载率:", z[0].load)
    print("gpu显存:", z[0].memoryTotal)
    print("gpu驱动:", z[0].driver)
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
class myCallback(tf.keras.callbacks.Callback):
    def __init__(self,path):
        self.fp = path

    def on_epoch_end(self, epoch, logs):
        print(epoch)
        #logs如下
        """
        {'loss': 0.35939720273017883, 'accuracy': 0.5134039521217346, 'val_loss': 0.2836693525314331, 'val_accuracy': 0.476856529712677}
        """
        """
        写入loss等基本信息
        epch,loss,accuracy,val_loss,val_accuracy
        """
        fp=open(self.fp, "a+")
        fp.write(str(epoch)+","+str(logs["loss"])+","+str(logs["accuracy"])+"\n")
        fp.close()
if __name__ == '__main__':
    if setting.is_only:
        main()
        sys.exit(0)
    if setting.is_vit:
        print("使用的多模态模型为 VILT")
    else:
        print("使用的模型组合为",setting.imageclf,"+",setting.txtclf,"+",setting.placeclf)

    dataset=Dataset()

    if setting.is_val:
        if setting.txtclf != 'bert':
            dataset_train, dataset_test, dataset_val = dataset.data_process_unbert()
        else:
            dataset_train, dataset_test, dataset_val=dataset.data_process()
    else:
        if setting.txtclf != 'bert':
            dataset_train, dataset_test = dataset.data_process_unbert()
        else:
            dataset_train, dataset_test=dataset.data_process()
    len_val_tokenize = dataset.len_val_tokenize
    len_test_tokenize = dataset.len_test_tokenize
    len_train_tokenize = dataset.len_train_tokenize
    class_num = int(dataset.class_num)
    print(class_num)
    vocab_size = dataset.vocab_size
    inputs = Input(shape=(224,224,3), batch_size=setting.batch)
    inputs2= tf.keras.layers.Input(shape=(len_train_tokenize),dtype=tf.int32,batch_size=setting.batch)
    inputs3 = tf.keras.layers.Input(shape=(len_train_tokenize), dtype=tf.int32, batch_size=setting.batch)
    inputs4 = tf.keras.layers.Input(shape=(len_train_tokenize), dtype=tf.int32, batch_size=setting.batch)
    inputs5 = tf.keras.layers.Input(shape=(2), dtype=tf.float32, batch_size=setting.batch)
    if setting.txtclf!="bert":
        multimodelclf=MultimodalConcatBertClf(len_dataset=len_train_tokenize,vocab_size=vocab_size,hide_size=setting.hide_size,class_num=class_num,isbn=setting.isbn)
    else:
        if setting.is_vit:
            pass
            #multimodelclf=ViLTransformer(num_heads=setting.num_heads,patch_size=setting.patch_size,embed_dim=setting.embed_dim,layer_length=setting.layer_length,num_classes=class_num,isencoder=False)
        else:
            multimodelclf = MultimodalConcatBertClf( hide_size=setting.hide_size, class_num=class_num,
                                                isbn=setting.isbn)

    if setting.txtclf!='bert':
        input={
            "place":inputs5,
            "img":inputs,
            "txt":inputs2,
        }
    else:
        input = {
            "place": inputs5,
            "img": inputs,
            "input_ids": inputs2,
            "token_type_ids": inputs3,
            "attention_mask": inputs4,
        }
    out = multimodelclf(input)
    print(out)
    model = Model(inputs=input, outputs=out, name='concat_bert-tf2')
    # model.load_weights("save/bert10.h5")
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08, clipnorm=1)
    train_loss = tf.keras.losses.CategoricalCrossentropy()
    train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
    callbacks = [
        # 模型保存
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(setting.save_path, setting.imageclf+setting.txtclf+"{epoch}.h5"),
            monitor="accuracy",
            save_weights_only=True,
            verbose=1,
            period=10,
            save_best_only=True,
            mode="max",

        ),
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                  patience=20,
        #                                  restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(
            log_dir=setting.log_dir, histogram_freq=0, write_graph=True, write_images=False,
            update_freq='epoch', profile_batch=2, embeddings_freq=0,
            embeddings_metadata=None,
        ),
        myCallback("log1.txt")
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00000001)
    ]
    model.compile(
        optimizer=optimizer,
        loss=train_loss,
        metrics=['accuracy']
    )

    model.fit(
        dataset_train,
        epochs=setting.epoch,
        batch_size=setting.batch,        # initial_epoch=setting.initial_epoch,
        # validation_data=dataset_val,
        callbacks=callbacks,
    )
    inputs = Input(shape=(224, 224, 3), batch_size=setting.batch)
    inputs2 = tf.keras.layers.Input(shape=(len_test_tokenize), dtype=tf.int32, batch_size=setting.batch)
    inputs3 = tf.keras.layers.Input(shape=(len_test_tokenize), dtype=tf.int32, batch_size=setting.batch)
    inputs4 = tf.keras.layers.Input(shape=(len_test_tokenize), dtype=tf.int32, batch_size=setting.batch)
    inputs5 = tf.keras.layers.Input(shape=(2), dtype=tf.int32, batch_size=setting.batch)
    if setting.txtclf != "bert":
        multimodelclf = MultimodalConcatBertClf(len_dataset=len_test_tokenize, vocab_size=vocab_size,
                                                hide_size=setting.hide_size, class_num=class_num, isbn=setting.isbn)
    else:
        multimodelclf = MultimodalConcatBertClf(hide_size=setting.hide_size, class_num=class_num,
                                                isbn=setting.isbn)
    if setting.txtclf != 'bert':
        input = {
            "place": inputs5,
            "img": inputs,
            "txt": inputs2,
        }
    else:
        input = {
            "place": inputs5,
            "img": inputs,
            "input_ids": inputs2,
            "token_type_ids": inputs3,
            "attention_mask": inputs4,
        }
    out = multimodelclf(input)
    model = Model(inputs=input, outputs=out, name='concat_bert-tf2_veal')
    list_dir=[]
    for i in os.listdir(setting.save_path):
        a=re.findall("\d+?\d*", i)
        list_dir.append(int(a[0]))

    model.load_weights(setting.save_path+"/"+ setting.imageclf,"+",setting.txtclf+str(max(list_dir))+".h5")
    model.compile(
        optimizer=optimizer,
        loss=train_loss,
        metrics=['accuracy']
    )
    model.evaluate(dataset_test,callbacks=callbacks,steps=1)