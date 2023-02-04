import tensorflow as tf
from tqdm import *

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer,BertModel,BertConfig
import numpy as np
import setting
import random
import cv2
class Dataset:
    def __init__(self):
        self.path_data=setting.path_data
        self.path_data_img=setting.path_img
        self.label=[]
        self.place_list=[]
        self.txt_list=[]
        self.img_list=[]
        self.img_data=[]
        self.place_data=[]
        self.bert_tokenize=[]
        self.dataset_len=None
        with open(self.path_data,"r",encoding="GBK") as fp:
            data=fp.read().split("\n")
            self.dataset_len=len(data)-1
            for i in data:
                if i=="":
                    continue
                dic=eval(i)
                self.label.append(dic["scenelabel"])
                self.place_list.append([float(dic["longitude"]),float(dic["latitude"])])
                self.txt_list.append(dic["describe"])
                self.img_list.append(dic["image"])
        assert self.dataset_len != None, "The dataset is empty, or there is a problem somewhere in the dataset"
        self.class_num = len(set(self.label))
        print(self.class_num)
        self.label=tf.one_hot(self.label,depth=len(set(self.label)))
        self.len_val_tokenize=None
        self.len_test_tokenize = None
        self.len_train_tokenize=None
        self.tokenizer = BertTokenizer.from_pretrained(setting.tokenize_path)
        self.vocab_size=len(self.tokenizer.vocab)
        self.len_dataset_noise=None
        # self.txt_data=self.tokenizer(self.txt_list, truncation=True, padding=True, max_length=setting.max_size)
        # self.input_ids=self.txt_data["input_ids"]
        # self.token_type_ids=self.txt_data["token_type_ids"]
        # self.attention_mask=self.txt_data["attention_mask"]
        # print(len(self.input_ids),len(self.token_type_ids),len(self.attention_mask))
    def __len__(self):
        return self.dataset_len
    def normailze(self,list_track):
        """
        :param list_track: 需要归一化的坐标
        :return: 列表
        """
        gps_x = np.array([i[0] for i in list_track], dtype=float)
        gps_y = np.array([i[1] for i in list_track], dtype=float)

        gps_x_max = float(max(gps_x))
        gps_y_max = float(max(gps_y))
        gps_x_min = float(min(gps_x))
        gps_y_min = float(min(gps_y))

        gps_x = (gps_x - gps_x_min) / (gps_x_max - gps_x_min)
        gps_y = (gps_y - gps_y_min) / (gps_y_max - gps_y_min)

        return [[i, j] for i, j in zip(gps_x.tolist(), gps_y.tolist())]
    def img_get(self,img_path):
        """
        :param img_path: 这个img_path只是个名字如：19在img_get这个函数中完成拼接等任务
        :return: 处理过的图片样式
        """
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
        return cv2.resize(image, (setting.img_w, setting.img_h))
    def image_numpy_deal(self,image):
        image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
        return cv2.resize(image, (setting.img_w, setting.img_h))
    def img_get_append_noise(self,img_path):
        """
        这个函数的作用是给函数添加噪声
        :param img_path: 这个img_path只是个名字如：19在img_get这个函数中完成拼接等任务
        :return: 处理过的图片样式
        """
        def append_noise(image,mode):
            def gasuss_noise(image, mean=0, var=0.001):
                '''
                手动添加高斯噪声
                mean : 均值
                var : 方差
                '''
                image = np.array(image / 255, dtype=float)
                noise = np.random.normal(mean, var ** 0.5, image.shape)  # 正态分布
                out = image + noise
                if out.min() < 0:
                    low_clip = -1.
                else:
                    low_clip = 0.
                out = np.clip(out, low_clip, 1.0)
                out = np.uint8(out * 255)
                return out

            def sp_noise(image, prob=0.1):
                '''
                手动添加椒盐噪声
                prob:噪声比例
                '''
                output = np.zeros(image.shape, np.uint8)
                thres = 1 - prob
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        rdn = random.random()
                        if rdn < prob:
                            output[i][j] = 0
                        elif rdn > thres:
                            output[i][j] = 255
                        else:
                            output[i][j] = image[i][j]
                return output

            if mode=="gasuss":
                return gasuss_noise(image,mean=setting.mean,var=setting.var)
            elif mode=="Impulse":
                return sp_noise(image,prob=setting.prob)
            else:
                assert "There is no corresponding noise handling device"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
        iamge=cv2.resize(image, (setting.img_w, setting.img_h))
        iamge=append_noise(iamge,mode=setting.noise_mode)
        return iamge
    def data_process(self):
        """

        :return: 返回处理后的数据集
        """
        self.place_data=self.normailze(self.place_list)
        flag=1
        for i in range(self.dataset_len):
            self.img_data.append(self.img_get(self.path_data_img+"/"+str(self.img_list[i])+".jpg")/255.0+1e-3)

        # data_list=[{
        #     "place": self.place_data[i],
        #     "img": self.img_data[i],
        #     "txt": {
        #         "input_ids":self.input_ids[i],
        #         "token_type_ids":self.token_type_ids[i],
        #         "attention_mask":self.attention_mask[i]
        #     }
        # } for i in range(self.dataset_len)]


        x_train_place,x_test_place,x_train_img_data,x_test_img_data,x_train_txt_list,x_test_txt_list,y_train,y_test= train_test_split(
            self.place_data,self.img_data,self.txt_list, self.label.numpy(), test_size=0.2)

        x_test_txt_list=self.tokenizer(x_test_txt_list,truncation=True, padding=True, max_length=setting.max_size)
        if setting.is_val:
            x_train_place, x_val_place, x_train_img_data, x_val_img_data, x_train_txt_list, x_val_txt_list, y_train, y_val = train_test_split(
               x_train_place, x_train_img_data, x_train_txt_list, y_train, test_size=0.2)

            x_val_txt_list = self.tokenizer(x_val_txt_list, truncation=True, padding=True,
                                              max_length=setting.max_size)
            self.len_val_tokenize=tf.convert_to_tensor(x_val_txt_list["input_ids"]).shape[1]
            dataset_val = tf.data.Dataset.from_tensor_slices(({
                                                                  "place":x_val_place,
                                                                  "img":x_val_img_data,
                                                                  "input_ids":x_val_txt_list["input_ids"],
                                                                  "token_type_ids": x_val_txt_list["token_type_ids"],
                                                                  "attention_mask":x_val_txt_list["attention_mask"]
                                                              }, y_val)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeeat)

            flag=0
        x_train_txt_list = self.tokenizer(x_train_txt_list, truncation=True, padding=True,
                                          max_length=setting.max_size)

        self.len_train_tokenize=tf.convert_to_tensor(x_train_txt_list["input_ids"]).shape[1]
        dataset_train = tf.data.Dataset.from_tensor_slices(({
                                                              "place": x_train_place,
                                                              "img": x_train_img_data,
                                                            "input_ids": x_train_txt_list["input_ids"],
                                                            "token_type_ids": x_train_txt_list["token_type_ids"],
                                                            "attention_mask": x_train_txt_list["attention_mask"]
                                                          }, y_train)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        self.len_test_tokenize=tf.convert_to_tensor(x_test_txt_list["input_ids"]).shape[1]
        dataset_test = tf.data.Dataset.from_tensor_slices(({
                                                                "place": x_test_place,
                                                                "img": x_test_img_data,
                                                               "input_ids": x_test_txt_list["input_ids"],
                                                               "token_type_ids": x_test_txt_list["token_type_ids"],
                                                               "attention_mask": x_test_txt_list["attention_mask"]
                                                            }, y_test)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)


        if setting.is_val:
            print("验证数据集的长度为:", len(dataset_val)*setting.batch)
            print(dataset_val)
            print("训练数据集长度为:", len(dataset_train)*setting.batch)
            print(dataset_train)
            print("测试数据集长度为:", len(dataset_test)*setting.batch)
            print(dataset_test)
            return dataset_train,dataset_test,dataset_val
        print("训练数据集长度为:", len(dataset_train)*setting.batch)
        print(dataset_train)
        print("测试数据集长度为:", len(dataset_test)*setting.batch)
        print(dataset_test)
        return dataset_train,dataset_test
    def data_process_unbert(self):
        """

        :return: 返回处理后的数据集
        """
        self.place_data=self.normailze(self.place_list)

        for i in range(self.dataset_len):
            self.img_data.append(self.img_get(self.path_data_img+"/"+str(self.img_list[i])+".jpg")/255.0+1e-3)

        # data_list=[{
        #     "place": self.place_data[i],
        #     "img": self.img_data[i],
        #     "txt": {
        #         "input_ids":self.input_ids[i],
        #         "token_type_ids":self.token_type_ids[i],
        #         "attention_mask":self.attention_mask[i]
        #     }
        # } for i in range(self.dataset_len)]


        x_train_place,x_test_place,x_train_img_data,x_test_img_data,x_train_txt_list,x_test_txt_list,y_train,y_test= train_test_split(
            self.place_data,self.img_data,self.txt_list, self.label.numpy(), test_size=0.2)

        x_test_txt_list=self.tokenizer(x_test_txt_list,truncation=True, padding=True, max_length=setting.max_size)
        if setting.is_val:
            x_train_place, x_val_place, x_train_img_data, x_val_img_data, x_train_txt_list, x_val_txt_list, y_train, y_val = train_test_split(
               x_train_place, x_train_img_data, x_train_txt_list, y_train, test_size=0.2)

            x_val_txt_list = self.tokenizer(x_val_txt_list, truncation=True, padding=True,
                                              max_length=setting.max_size)
            self.len_val_tokenize=tf.convert_to_tensor(x_val_txt_list["input_ids"]).shape[1]
            dataset_val = tf.data.Dataset.from_tensor_slices(({
                                                                  "place":x_val_place,
                                                                  "img":x_val_img_data,
                                                                  "txt":x_val_txt_list["input_ids"],
                                                              }, y_val)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)


        x_train_txt_list = self.tokenizer(x_train_txt_list, truncation=True, padding=True,
                                          max_length=setting.max_size)

        self.len_train_tokenize=tf.convert_to_tensor(x_train_txt_list["input_ids"]).shape[1]
        dataset_train = tf.data.Dataset.from_tensor_slices(({
                                                              "place": x_train_place,
                                                              "img": x_train_img_data,
                                                            "txt": x_train_txt_list["input_ids"]
                                                          }, y_train)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        self.len_test_tokenize=tf.convert_to_tensor(x_test_txt_list["input_ids"]).shape[1]
        dataset_test = tf.data.Dataset.from_tensor_slices(({
                                                                "place": x_test_place,
                                                                "img": x_test_img_data,
                                                               "txt": x_test_txt_list["input_ids"],
                                                            }, y_test)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)


        if setting.is_val:
            print("验证数据集的长度为:", len(dataset_val)*setting.batch)
            print(dataset_val)
            print("训练数据集长度为:", len(dataset_train)*setting.batch)
            print(dataset_train)
            print("测试数据集长度为:", len(dataset_test)*setting.batch)
            print(dataset_test)
            return dataset_train,dataset_test,dataset_val
        print("训练数据集长度为:", len(dataset_train)*setting.batch)
        print(dataset_train)
        print("测试数据集长度为:", len(dataset_test)*setting.batch)
        print(dataset_test)
        return dataset_train,dataset_test
    def data_process_img_only(self):
        for i in range(self.dataset_len):
            self.img_data.append(self.img_get(self.path_data_img + "/" + str(self.img_list[i]) + ".jpg") / 255.0 + 1e-3)
        x_train_img_data, x_test_img_data,y_train, y_test = train_test_split(
           self.img_data, self.label.numpy(), test_size=0.2)
        if setting.is_val:
            x_train_img_data, x_val_img_data,y_train, y_val = train_test_split(
                x_train_img_data, y_train, test_size=0.2)
            dataset_val = tf.data.Dataset.from_tensor_slices((x_val_img_data, y_val)).shuffle(setting.BUFFER_SIZE).prefetch(
                tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        dataset_train = tf.data.Dataset.from_tensor_slices((x_train_img_data, y_train)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        dataset_test = tf.data.Dataset.from_tensor_slices((x_test_img_data, y_test)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        if setting.is_val:
            print("验证数据集的长度为:", len(dataset_val) * setting.batch)
            print(dataset_val)
            print("训练数据集长度为:", len(dataset_train) * setting.batch)
            print(dataset_train)
            print("测试数据集长度为:", len(dataset_test) * setting.batch)
            print(dataset_test)
            return dataset_train, dataset_test, dataset_val
        print("训练数据集长度为:", len(dataset_train) * setting.batch)
        print(dataset_train)
        print("测试数据集长度为:", len(dataset_test) * setting.batch)
        print(dataset_test)
        return dataset_train, dataset_test
    def data_process_txt_only(self):

        x_train_txt_data, x_test_txt_data, y_train, y_test = train_test_split(
            self.txt_list, self.label.numpy(), test_size=0.2)
        x_test_txt_list = self.tokenizer(x_test_txt_data, truncation=True, padding=True, max_length=setting.max_size)
        if setting.is_val:
            x_train_txt_data, x_val_txt_data, y_train, y_val = train_test_split(
                x_train_txt_data, y_train, test_size=0.2)
            x_val_txt_list = self.tokenizer(x_val_txt_data, truncation=True, padding=True,
                                            max_length=setting.max_size)
            self.len_val_tokenize = tf.convert_to_tensor(x_val_txt_list["input_ids"]).shape[1]
            dataset_val = tf.data.Dataset.from_tensor_slices(({
                                                                  "input_ids": x_val_txt_list["input_ids"],
                                                                  "token_type_ids": x_val_txt_list["token_type_ids"],
                                                                  "attention_mask": x_val_txt_list["attention_mask"]
                                                              }, y_val)).shuffle(setting.BUFFER_SIZE).prefetch(
                tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        x_train_txt_list = self.tokenizer(x_train_txt_data, truncation=True, padding=True,
                                          max_length=setting.max_size)
        self.len_train_tokenize=tf.convert_to_tensor(x_train_txt_list["input_ids"]).shape[1]
        dataset_train = tf.data.Dataset.from_tensor_slices(({
                                                                "input_ids": x_train_txt_list["input_ids"],
                                                                "token_type_ids": x_train_txt_list["token_type_ids"],
                                                                "attention_mask": x_train_txt_list["attention_mask"]
                                                            }, y_train)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        self.len_test_tokenize = tf.convert_to_tensor(x_test_txt_list["input_ids"]).shape[1]
        dataset_test = tf.data.Dataset.from_tensor_slices(({
                                                               "input_ids": x_test_txt_list["input_ids"],
                                                               "token_type_ids": x_test_txt_list["token_type_ids"],
                                                               "attention_mask": x_test_txt_list["attention_mask"]
                                                           }, y_test)).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        if setting.is_val:
            print("验证数据集的长度为:", len(dataset_val) * setting.batch)
            print(dataset_val)
            print("训练数据集长度为:", len(dataset_train) * setting.batch)
            print(dataset_train)
            print("测试数据集长度为:", len(dataset_test) * setting.batch)
            print(dataset_test)
            return dataset_train, dataset_test, dataset_val
        print("训练数据集长度为:", len(dataset_train) * setting.batch)
        print(dataset_train)
        print("测试数据集长度为:", len(dataset_test) * setting.batch)
        print(dataset_test)
        return dataset_train, dataset_test

    def data_process_noise(self):
        """

        :return: 返回处理后的数据集
        """

        self.place_data = self.normailze(self.place_list)[:600]
        text_noise=[]
        if setting.txt_mask!=0:
            for i in self.txt_list:
                str1=""
                for k in i:
                    if 0<=random.randint(0,10)<int(setting.txt_mask)*10:
                        str1+="[EMP]"
                    else:
                        str1+=k
                text_noise.append(str1)
        else:
            text_noise=self.txt_list
        flag = 1
        for i in range(600):
           # print(self.path_data_img + "/" + str(self.img_list[i]) + ".jpg")
            self.img_data.append(self.img_get_append_noise(self.path_data_img + "/" + str(self.img_list[i]) + ".jpg") / 255.0 + 1e-3)

        # data_list=[{
        #     "place": self.place_data[i],
        #     "img": self.img_data[i],
        #     "txt": {
        #         "input_ids":self.input_ids[i],
        #         "token_type_ids":self.token_type_ids[i],
        #         "attention_mask":self.attention_mask[i]
        #     }
        # } for i in range(self.dataset_len)]

        x_test_txt_list=self.tokenizer(text_noise[:600], truncation=True, padding=True, max_length=setting.max_size)
        # x_train_place, x_test_place, x_train_img_data, x_test_img_data, x_train_txt_list, x_test_txt_list, y_train, y_test = train_test_split(
        #     self.place_data, self.img_data, self.txt_list, self.label.numpy(), test_size=0.2)
        #
        # x_test_txt_list = self.tokenizer(x_test_txt_list, truncation=True, padding=True, max_length=setting.max_size)
        # if setting.is_val:
        #     x_train_place, x_val_place, x_train_img_data, x_val_img_data, x_train_txt_list, x_val_txt_list, y_train, y_val = train_test_split(
        #         x_train_place, x_train_img_data, x_train_txt_list, y_train, test_size=0.2)
        #
        #     x_val_txt_list = self.tokenizer(x_val_txt_list, truncation=True, padding=True,
        #                                     max_length=setting.max_size)
        #     self.len_val_tokenize = tf.convert_to_tensor(x_val_txt_list["input_ids"]).shape[1]
        #     dataset_val = tf.data.Dataset.from_tensor_slices(({
        #                                                           "place": x_val_place,
        #                                                           "img": x_val_img_data,
        #                                                           "input_ids": x_val_txt_list["input_ids"],
        #                                                           "token_type_ids": x_val_txt_list["token_type_ids"],
        #                                                           "attention_mask": x_val_txt_list["attention_mask"]
        #                                                       }, y_val)).shuffle(setting.BUFFER_SIZE).prefetch(
        #         tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeeat)
        #
        #     flag = 0
        # x_train_txt_list = self.tokenizer(x_train_txt_list, truncation=True, padding=True,
        #                                   max_length=setting.max_size)
        #
        # self.len_train_tokenize = tf.convert_to_tensor(x_train_txt_list["input_ids"]).shape[1]
        # dataset_train = tf.data.Dataset.from_tensor_slices(({
        #                                                         "place": x_train_place,
        #                                                         "img": x_train_img_data,
        #                                                         "input_ids": x_train_txt_list["input_ids"],
        #                                                         "token_type_ids": x_train_txt_list["token_type_ids"],
        #                                                         "attention_mask": x_train_txt_list["attention_mask"]
        #                                                     }, y_train)).shuffle(setting.BUFFER_SIZE).prefetch(
        #     tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        # self.len_test_tokenize = tf.convert_to_tensor(x_test_txt_list["input_ids"]).shape[1]
        # self.len_test_tokenize = tf.convert_to_tensor(x_test_txt_list["input_ids"]).shape[1]
        self.len_dataset_noise=tf.convert_to_tensor(x_test_txt_list["input_ids"]).shape[1]
        dataset = tf.data.Dataset.from_tensor_slices(({
                                                               "place": self.place_data,
                                                               "img": self.img_data,
                                                               "input_ids": x_test_txt_list["input_ids"],
                                                               "token_type_ids": x_test_txt_list["token_type_ids"],
                                                               "attention_mask": x_test_txt_list["attention_mask"]
                                                           }, self.label.numpy()[:600])).shuffle(setting.BUFFER_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE).batch(setting.batch).repeat(setting.repeat)
        #
        # if setting.is_val:
        #     print("验证数据集的长度为:", len(dataset_val) * setting.batch)
        #     print(dataset_val)
        #     print("训练数据集长度为:", len(dataset_train) * setting.batch)
        #     print(dataset_train)
        #     print("测试数据集长度为:", len(dataset_test) * setting.batch)
        #     print(dataset_test)
        #     return dataset_train, dataset_test, dataset_val
        print("**加入噪声的参数为**")
        print("gassus:",setting.mean,setting.var)
        print("imp",setting.prob)
        print("mask",setting.txt_mask)
        print("加入噪声的训练集为：:", len(dataset) * setting.batch)
        print(dataset)
        return dataset
    def get_one_dataset(self,poition,txt,image):

        point=[i/100 for i in poition]
        text=txt
        if len(txt)<setting.sentense_max_size:
            for i in range(setting.sentense_max_size-len(txt)):
                text+="[PAD]"
        text=self.tokenizer(text)
        image=self.image_numpy_deal(image)/255.0+1e-3
        # dataset_val = tf.data.Dataset.from_tensor_slices(({
        #                                                       "place": point,
        #                                                       "img":image,
        #                                                       "input_ids": x_val_txt_list["input_ids"],
        #                                                       "token_type_ids": x_val_txt_list["token_type_ids"],
        #                                                       "attention_mask": x_val_txt_list["attention_mask"]
        #                                                   }, y_val))
        return point,text,image
# # # data=Dataset()
# data.img_get_append_noise()