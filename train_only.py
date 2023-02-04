import os
import setting
import re
if setting.is_gpu==False:

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from model.bert import Bertclf
from model.image import imageclf as resnet_model
from data import Dataset
if setting.is_gpu:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

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



def main():
    if setting.is_txt:
        print("模型组合为:","only txt")
    elif setting.is_img:
        print("模型组合为:", "only img")
    dataset = Dataset()
    class_num = dataset.class_num
    vocab_size = dataset.vocab_size
    if setting.is_val:
        if setting.is_txt:
            dataset_train, dataset_test, dataset_val = dataset.data_process_txt_only()
        else:
            dataset_train, dataset_test, dataset_val = dataset.data_process_img_only()
    else:
        if setting.is_txt:
            dataset_train, dataset_test = dataset.data_process_txt_only()
        else:
            dataset_train, dataset_test = dataset.data_process_img_only()
    len_val_tokenize = dataset.len_val_tokenize
    len_test_tokenize = dataset.len_test_tokenize
    len_train_tokenize = dataset.len_train_tokenize
    if setting.is_txt:
        inputs = tf.keras.layers.Input(shape=(len_train_tokenize),dtype=tf.int32)
        inputs1 = tf.keras.layers.Input(shape=(len_train_tokenize), dtype=tf.int32)
        inputs2 = tf.keras.layers.Input(shape=(len_train_tokenize), dtype=tf.int32)
        bert=Bertclf(setting.bert_path,setting.initializer_range,class_nums=class_num)
        input={"input_ids":inputs,
                     "token_type_ids":inputs1,
                     "attention_mask":inputs2
                     }
        out = bert(input)
        model = tf.keras.Model(inputs=input, outputs=out, name='bert-tf2')
        model.summary()
        optimizer =tf.keras.optimizers.Adam(learning_rate=1e-5,epsilon=1e-08, clipnorm=1)
        train_loss = tf.keras.losses.CategoricalCrossentropy()
        train_accuracy =tf.keras.metrics.Accuracy(name='train_accuracy')

        callbacks = [
            # 模型保存
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(setting.save_path, "bert{epoch}.h5"),
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

        # model.fit(
        #     dataset_train,
        #     epochs=setting.epoch,
        #     batch_size=setting.batch,  # initial_epoch=setting.initial_epoch,
        #     # validation_data=dataset_val,
        #     callbacks=callbacks,
        # )

        inputs2 = tf.keras.layers.Input(shape=(len_test_tokenize), dtype=tf.int32, batch_size=setting.batch)
        inputs3 = tf.keras.layers.Input(shape=(len_test_tokenize), dtype=tf.int32, batch_size=setting.batch)
        inputs4 = tf.keras.layers.Input(shape=(len_test_tokenize), dtype=tf.int32, batch_size=setting.batch)

        input = {

            "input_ids": inputs2,
            "token_type_ids": inputs3,
            "attention_mask": inputs4,
        }
        bert=Bertclf(setting.bert_path,setting.initializer_range,class_nums=class_num,isencoder=False)
        out = bert(input)
        model = tf.keras.Model(inputs=input, outputs=out, name='concat_bert-tf2_veal')
        list_dir = []
        for i in os.listdir(setting.save_path):
            a = re.findall("\d+?\d*", i)

            list_dir.append(int(a[0]))

       # model.load_weights(setting.save_path + "/bert" + str(max(list_dir)) + ".h5")
        model.compile(
            optimizer=optimizer,
            loss=train_loss,
            metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall'), ]
        )
        model.load_weights("test_model/only_bert.h5")
        model.evaluate(dataset_test)
    elif setting.is_img:
        inputs = tf.keras.layers.Input(shape=(224, 224, 3), batch_size=setting.batch)
        imageclf=resnet_model(class_num=class_num,num_image_embeds=setting.num_image_embeds,pool_func=setting.pool_func,isencoder=False)
        out = imageclf(inputs)
        model =  tf.keras.Model(inputs=inputs, outputs=out, name='resnet-tf2')
        model.summary()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08, clipnorm=1)
        train_loss = tf.keras.losses.CategoricalCrossentropy()
        train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

        callbacks = [
            # 模型保存
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(setting.save_path, "resnet{epoch}.h5"),
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
            metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall'), ]
        )
        model.load_weights("test_model/only_resnet.h5")
        # model.fit(
        #     dataset_train,
        #     epochs=setting.epoch,
        #     batch_size=setting.batch,  # initial_epoch=setting.initial_epoch,
        #     # validation_data=dataset_val,
        #     callbacks=callbacks,
        # )
        model.evaluate(dataset_train)