import os
import sys
from os.path import join

sys.path.append(os.pardir)

import random
import re
from collections import Counter
from copy import copy, deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,precision_score,roc_curve,auc,average_precision_score,log_loss
from copy import deepcopy, copy
from collections import Counter

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)
from torchvision import datasets


tp = transforms.ToTensor()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
     ])
transform_fn = transforms.Compose([
    transforms.ToTensor()
])

from utils.basic_functions import (fetch_data_and_label,
                                   get_class_i, get_labeled_data,
                                   label_to_one_hot)

# DATA_PATH ='./load/share_dataset/'  #'../../../share_dataset/'
DATA_PATH ='load/share_dataset/'
IMAGE_DATA = ['mnist', 'cifar10', 'cifar100', 'cifar20', 'utkface', 'facescrub', 'places365']
TABULAR_DATA = ['breast_cancer_diagnose','diabetes','adult_income','criteo','credit','nursery','avazu']
GRAPH_DATA = ['cora']
TEXT_DATA = ['news20']


def dataset_partition(args, index, dst, half_dim):
    if args.case[:12] == 'single_party':
        return (dst[0][:, args.data_split[0]:args.data_split[1]], dst[1])
    if args.k == 1:
        return dst
    if args.dataset.dataset_name in IMAGE_DATA:
        if len(dst) == 2: # IMAGE_DATA without attribute
            if args.k == 2:
                if index == 0:
                    return (dst[0][:, :, :half_dim, :], None)
                elif index == 1:
                    return (dst[0][:, :, half_dim:, :], dst[1])
                else:
                    assert index <= 1, "invalide party index"
                    return None
            elif args.k == 4:
                if index == 3:
                    return (dst[0][:, :, half_dim:, half_dim:], dst[1])
                else:
                    if index == 0:
                        return (dst[0][:, :, :half_dim, :half_dim], None)
                    elif index == 1:
                        return (dst[0][:, :, :half_dim, half_dim:], None)
                    elif index == 2:
                        return (dst[0][:, :, half_dim:, :half_dim], None)
                    else:
                        assert index <= 3, "invalide party index"
                        return None
            elif args.k == 1: # Centralized Training
                return (dst[0], dst[1])
            else:
                assert (args.k == 2 or args.k == 4), "total number of parties not supported for data partitioning"
                return None
        elif len(dst) == 3: # IMAGE_DATA with attribute
            if args.k == 2:
                if index == 0:
                    return (dst[0][:, :, :half_dim, :], None, dst[2])
                    # return (dst[0][:, :, half_dim:, :], None, None)
                elif index == 1:
                    return (dst[0][:, :, half_dim:, :], dst[1], dst[2])
                    # return (dst[0][:, :, :half_dim, :], dst[1], dst[2])
                else:
                    assert index <= 1, "invalide party index"
                    return None
            elif args.k == 4:
                if index == 3:
                    return (dst[0][:, :, half_dim:, half_dim:], dst[1], dst[2])
                else:
                    # passive party does not have label
                    if index == 0:
                        return (dst[0][:, :, :half_dim, :half_dim], None, dst[2])
                    elif index == 1:
                        return (dst[0][:, :, :half_dim, half_dim:], None, dst[2])
                    elif index == 2:
                        return (dst[0][:, :, half_dim:, :half_dim], None, dst[2])
                    else:
                        assert index <= 3, "invalide party index"
                        return None
            elif args.k == 1: # Centralized Training
                return (dst[0], dst[1], dst[2])
            else:
                assert (args.k == 2 or args.k == 4), "total number of parties not supported for data partitioning"
                return None
    elif args.dataset.dataset_name in ['nuswide']:
        if args.k == 2:
            if index == 0:
                return (dst[0][0],None) # passive party with text
            else:
                return (dst[0][1], dst[1]) # active party with image
        else:
            assert (args.k == 2), "total number of parties not supported for data partitioning"
            return None
    elif args.dataset.dataset_name in TABULAR_DATA:
        dim_list=[]
        for ik in range(args.k):
            dim_list.append(int(args.model_list[ik]['input_dim']))
            if len(dim_list)>1:
                for i in range(1, len(dim_list)):
                    dim_list[i]=dim_list[i]+dim_list[i-1]
        dim_list.insert(0,0)

        if index == (args.k-1):
            return (dst[0][:, dim_list[index]:], dst[1])
        else:
            # passive party does not have label
            if index <= (args.k-1):  
                return (dst[0][:, dim_list[index]:dim_list[index+1]], None)
            else:
                assert index <= (args.k-1), "invalide party index"
                return None
    elif args.dataset.dataset_name in TEXT_DATA: 
        dim_list=[]
        for ik in range(args.k):
            dim_list.append(int(args.model_list[str(ik)]['input_dim']))
            if len(dim_list)>1:
                dim_list[-1]=dim_list[-1]+dim_list[-2]
        dim_list.insert(0,0)
        
        if args.k == 1:
            return (dst[0], dst[1])

        if index == (args.k-1):
            return (dst[0][:, dim_list[index]:], dst[1])
        else:
            # passive party does not have label
            if index <= (args.k-1):  
                return (dst[0][:, dim_list[index]:dim_list[index+1]], None)
            else:
                assert index <= (args.k-1), "invalide party index"
                return None
    
def load_dataset_per_party(args, index):
    print('load_dataset_per_party')

    half_dim = -1

    if args.dataset.dataset_name == "cifar100":
        half_dim = 16
        train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.dataset.num_classes)
        train_dst = (torch.tensor(data), label)

        test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.dataset.num_classes)
        test_dst = (torch.tensor(data), label)
    elif args.dataset.dataset_name == "cifar20":
        half_dim = 16
        train_dst = datasets.CIFAR100(DATA_PATH, download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.dataset.num_classes)
        train_dst = (torch.tensor(data), label)
        
        test_dst = datasets.CIFAR100(DATA_PATH, download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.dataset.num_classes)
        test_dst = (torch.tensor(data), label)
    elif args.dataset.dataset_name == "cifar10":
        half_dim = 16
        train_dst = datasets.CIFAR10(DATA_PATH, download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.dataset.num_classes)
        train_dst = (torch.tensor(data), label)

        test_dst = datasets.CIFAR10(DATA_PATH, download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.dataset.num_classes)
        test_dst = (torch.tensor(data), label)
    elif args.dataset.dataset_name == "mnist":
        half_dim = 14
        train_dst = datasets.MNIST("~/.torch", download=True, train=True, transform=transform_fn)
        data, label = fetch_data_and_label(train_dst, args.dataset.num_classes)
        train_dst = (torch.tensor(data), label)

        test_dst = datasets.MNIST("~/.torch", download=True, train=False, transform=transform_fn)
        data, label = fetch_data_and_label(test_dst, args.dataset.num_classes)
        test_dst = (data, label)
    elif args.dataset.dataset_name == 'utkface': # with attribute
        # 0.8 for train (all for train, but with 50% also for aux) and 0.2 for test
        half_dim = 25
        with np.load(DATA_PATH + 'UTKFace/utk_resize.npz') as f:
            data = f['imgs']
            # 'gender'=2, 'age'=11(after binning), 'race'=5
            label = f['gender' + 's']
            attribute = f['race' + 's']
            # attribute = f['age' + 's']
            # def binning_ages(a):
            #     buckets = [5, 10, 18, 25, 30, 35, 45, 55, 65, 75]
            #     for i, b in enumerate(buckets):
            #         if a <= b:
            #             return i
            #     return len(buckets)
            # attribute = [binning_ages(age) for age in attribute]
            # print(np.mean(data[:, :, :, 0]), np.mean(data[:, :, :, 1]), np.mean(data[:, :, :, 2]))
            # print(np.std(data[:, :, :, 0]), np.std(data[:, :, :, 1]), np.std(data[:, :, :, 2]))
            # MEANS = [152.13768243, 116.5061518, 99.7395918]
            # STDS = [65.71289385, 58.56545956, 57.4306078]
            MEANS = [137.10815842537994, 121.46186260277386, 112.96171130304792]
            STDS = [76.95932152349954, 74.33070450734535, 75.40728437766884]
            def channel_normalize(x):
                x = np.asarray(x, dtype=np.float32)
                x = x / 255.0
                # x[:, :, :, 0] = (x[:, :, :, 0] - MEANS[0]) / STDS[0]
                # x[:, :, :, 1] = (x[:, :, :, 1] - MEANS[1]) / STDS[1]
                # x[:, :, :, 2] = (x[:, :, :, 2] - MEANS[2]) / STDS[2]
                return x
            data = channel_normalize(data)
            label = np.asarray(label, dtype=np.int32)
            attribute = np.asarray(attribute, dtype=np.int32)
            X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(data, label, attribute, train_size=0.8, stratify=attribute, random_state=args.current_seed)
            # [debug] in load dataset for utkface, X_aux.shape=torch.Size([9482, 50, 50, 3]), y_aux.shape=torch.Size([9482]), a_aux.shape=torch.Size([9482])
            # [debug] in load dataset for utkface, X_train.shape=torch.Size([18964, 50, 50, 3]), y_train.shape=(18964,), a_train.shape=(18964,)
            # [debug] in load dataset for utkface, X_test.shape=torch.Size([4741, 50, 50, 3]), y_test.shape=(4741,), a_test.shape=(4741,)
            # [debug] in load dataset, number of attributes for UTKFace: 5
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            # print(f"[debug] in load dataset for utkface, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}, a_train.shape={a_train.shape}")
            # print(f"[debug] in load dataset for utkface, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}, a_test.shape={a_test.shape}")
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
            a_train = torch.tensor(a_train, dtype=torch.long)
            a_test = torch.tensor(a_test, dtype=torch.long)
            train_dst = (X_train, y_train, a_train)
            test_dst = (X_test, y_test, a_test)
            args.num_attributes = len(np.unique(a_train.numpy()))
    elif args.dataset.dataset_name == 'facescrub':
        half_dim = 25
        def load_gender():
            i = 0
            name_gender = dict()
            for f in [DATA_PATH + 'FaceScrub/facescrub_actors.txt', DATA_PATH + 'FaceScrub/facescrub_actresses.txt']:
                with open(f) as fd:
                    fd.readline()
                    names = []
                    for line in fd.readlines():
                        components = line.split('\t')
                        assert (len(components) == 6)
                        name = components[0]  # .decode('utf8')
                        names.append(name)
                    name_gender.update(dict(zip(names, np.ones(len(names)) * i)))
                i += 1
            return name_gender
        with np.load(DATA_PATH + 'FaceScrub/Data/facescrub.npz') as f:
            data, attribute, names = [f['arr_%d' % i] for i in range(len(f.files))]

            name_gender = load_gender()
            label = [name_gender[names[i]] for i in attribute]
            label = np.asarray(label, dtype=np.int32)
            attribute = np.asarray(attribute, dtype=np.int32)
            if len(np.unique(attribute)) > 300: # only use the most common 500 person
                id_cnt = Counter(attribute)
                attribute_selected = [tup[0] for tup in id_cnt.most_common(300)]
                indices = []
                new_attribute = []
                all_indices = np.arange(len(attribute))
                for i, face_id in enumerate(attribute_selected):
                    face_indices = all_indices[attribute == face_id]
                    new_attribute.append(np.ones_like(face_indices) * i)
                    indices.append(face_indices)
                indices = np.concatenate(indices)
                data = data[indices]
                label = label[indices]
                attribute = np.concatenate(new_attribute)
                attribute = np.asarray(attribute, dtype=np.int32)
            # print(Counter(attribute).most_common()
            X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(data, label, attribute, train_size=0.8, stratify=attribute, random_state=args.current_seed)
            # Majority prop 0=0.5407%
            # [debug] in load dataset for FaceScrub, X_aux.shape=torch.Size([9062, 50, 50, 3]), y_aux.shape=torch.Size([9062]), a_aux.shape=torch.Size([9062])
            # [debug] in load dataset for FaceScrub, X_train.shape=torch.Size([18124, 50, 50, 3]), y_train.shape=(18124,), a_train.shape=(18124,)
            # [debug] in load dataset for FaceScrub, X_test.shape=torch.Size([4532, 50, 50, 3]), y_test.shape=(4532,), a_test.shape=(4532,)
            # [debug] in load dataset, number of attributes for FaceScrub: 300
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            print(f"[debug] in load dataset for FaceScrub, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}, a_train.shape={a_train.shape}")
            print(f"[debug] in load dataset for FaceScrub, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}, a_test.shape={a_test.shape}")
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
            a_train = torch.tensor(a_train, dtype=torch.long)
            a_test = torch.tensor(a_test, dtype=torch.long)
            train_dst = (X_train, y_train, a_train)
            test_dst = (X_test, y_test, a_test)
            args.num_attributes = len(np.unique(a_train.numpy()))
            print(f"[debug] in load dataset, number of attributes for FaceScrub: {args.num_attributes}")
    elif args.dataset.dataset_name == 'places365':
        half_dim = 64
        with np.load(DATA_PATH + 'Places365/place128.npz') as f:
            data, label, attribute = f['arr_0'], f['arr_1'], f['arr_2']
            unique_p = np.unique(attribute)
            p_to_id = dict(zip(unique_p, range(len(unique_p))))
            attribute = np.asarray([p_to_id[a] for a in attribute], dtype=np.int32)
            label = label.astype(np.int32)
            data = data / 255.0
            X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(data, label, attribute, train_size=0.8, stratify=attribute, random_state=args.current_seed)
            # [debug] in load dataset for places365, X_aux.shape=torch.Size([29200, 128, 128, 3]), y_aux.shape=torch.Size([29200]), a_aux.shape=torch.Size([29200])
            # [debug] in load dataset for places365, X_train.shape=torch.Size([58400, 128, 128, 3]), y_train.shape=(58400,), a_train.shape=(58400,)
            # [debug] in load dataset for places365, X_test.shape=torch.Size([14600, 128, 128, 3]), y_test.shape=(14600,), a_test.shape=(14600,)
            # [debug] in load dataset, number of attributes for Places365: 365
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            # print(f"[debug] in load dataset for places365, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}, a_train.shape={a_train.shape}")
            # print(f"[debug] in load dataset for places365, X_test.shape={X_test.shape}, y_test.shape={y_test.shape}, a_test.shape={a_test.shape}")
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
            a_train = torch.tensor(a_train, dtype=torch.long)
            a_test = torch.tensor(a_test, dtype=torch.long)
            train_dst = (X_train, y_train, a_train)
            test_dst = (X_test, y_test, a_test)
            args.num_attributes = len(np.unique(a_train.numpy()))
            # print(f"[debug] in load dataset, number of attributes for Places365: {args.num_attributes}")
    elif args.dataset.dataset_name == 'nuswide':
        half_dim = [1000, 634]
        if args.dataset.num_classes == 5:
            selected_labels = ['buildings', 'grass', 'animal', 'water', 'person'] # class_num = 5
        elif args.dataset.num_classes == 2:
            selected_labels = ['clouds','person'] # class_num = 2
            # sky 34969 light 21022
            # nature 34894 sunset 20757
            # water 31921 sea 17722
            # blue 31496 white 16938
            # clouds 26906 people 16077
            # bravo 26624 night 16057
            # landscape 23024 beach 15677
            # green 22625 architecture 15264
            # red 21983 art 14395
            # explore 21037 travel 13999

        # X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 60, 'Train')
        X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 60000, 'Train')
        
        data = [torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_image, dtype=torch.float32)]
        label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
        label = label_to_one_hot(label, num_classes=args.dataset.num_classes)
            
        train_dst = (data, label) # (torch.tensor(data),label)
        print("nuswide dataset [train]:", data[0].shape, data[1].shape, label.shape)
        # X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 40, 'Test')
        X_image, X_text, Y = get_labeled_data(DATA_PATH+'NUS_WIDE', selected_labels, 40000, 'Test')
        data = [torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_image, dtype=torch.float32)]
        label = torch.squeeze(torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long))
        label = label_to_one_hot(label, num_classes=args.dataset.num_classes)
        test_dst = (data, label)
        print("nuswide dataset [test]:", data[0].shape, data[1].shape, label.shape)

        data = data.rename(columns={data.columns[0]: 'POI'})
        

    elif args.dataset.dataset_name in TABULAR_DATA:
        if args.dataset.dataset_name == 'breast_cancer_diagnose':
            half_dim = 15
            df = pd.read_csv(DATA_PATH+"BreastCancer/wdbc.data",header = 0)
            X = df.iloc[:, 2:].values
            y = df.iloc[:, 1].values
            y = np.where(y=='B',0,1)
            y = np.squeeze(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=args.current_seed)
            
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        elif args.dataset.dataset_name == 'diabetes':
            half_dim = 4
            df = pd.read_csv(DATA_PATH+"Diabetes/diabetes.csv",header = 0)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=args.current_seed)
        elif args.dataset.dataset_name == 'adult_income':
            df = pd.read_csv(DATA_PATH+"Income/adult.csv",header = 0)
            df = df.drop_duplicates()
            # 'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
            # 'marital-status', 'occupation', 'relationship', 'race', 'gender',
            # 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            # 'income'
            # category_columns_index = [1,3,5,6,7,8,9,13]
            # num_category_of_each_column = [9,16,7,15,6,5,2,42]
            category_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender','native-country']
            for _column in category_columns:
                # Get one hot encoding of columns B
                one_hot = pd.get_dummies(df[_column], prefix=_column)
                # Drop column B as it is now encoded
                df = df.drop(_column,axis = 1)
                # Join the encoded df
                df = df.join(one_hot)
            y = df['income'].values
            y = np.where(y=='<=50K',0,1)
            df = df.drop('income',axis=1)
            X = df.values
            half_dim = 6+9 #=15 acc=0.83
            # half_dim = 6+9+16+7+15 #=53 acc=0.77
            # half_dim = int(X.shape[1]//2) acc=0.77
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=args.current_seed)
        elif args.dataset.dataset_name == 'criteo':
            df = pd.read_csv(DATA_PATH+"Criteo/train.txt", sep='\t', header=None)
            df = df.sample(frac=0.02, replace=False, random_state=42)
            df.columns = ["labels"] + ["I%d"%i for i in range(1,14)] + ["C%d"%i for i in range(14,40)]
            print("criteo dataset loaded")
            y = df["labels"].values
            X_p =  [col for col in df.columns if col.startswith('I')]
            X_a = [col for col in df.columns if col.startswith('C')]
            X_p = process_dense_feats(df, X_p)
            X_a = process_sparse_feats(df, X_a)
            print('X_p shape',X_p.shape)
            print('X_a shape',X_a.shape)
            X = pd.concat([X_a, X_p], axis=1).values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False)
        elif args.dataset.dataset_name == "credit":
            df = pd.read_csv(DATA_PATH+"tabledata/UCI_Credit_Card.csv")
            print("credit dataset loaded")

            X = df[
                [
                    "LIMIT_BAL",
                    "SEX",
                    "EDUCATION",
                    "MARRIAGE",
                    "AGE",
                    "PAY_0",
                    "PAY_2",
                    "PAY_3",
                    "PAY_4",
                    "PAY_5",
                    "PAY_6",
                    "BILL_AMT1",
                    "BILL_AMT2",
                    "BILL_AMT3",
                    "BILL_AMT4",
                    "BILL_AMT5",
                    "BILL_AMT6",
                    "PAY_AMT1",
                    "PAY_AMT2",
                    "PAY_AMT3",
                    "PAY_AMT4",
                    "PAY_AMT5",
                    "PAY_AMT6",
                ]
            ].values
            y = df["default.payment.next.month"].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.runtime.seed, stratify=y)
            scaler = StandardScaler() # MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif args.dataset.dataset_name == "nursery":
            df = pd.read_csv(DATA_PATH+"tabledata/nursery.data", header=None)
            print("nursery dataset loaded")
            df[8] = LabelEncoder().fit_transform(df[8].values)
            X_d = df.drop(8, axis=1)
            X_a = pd.get_dummies(
                X_d[X_d.columns[: int(len(X_d.columns) / 2)]], drop_first=True, dtype=int
            )
            print('X_a',X_a.shape)
            X_p = pd.get_dummies(
                X_d[X_d.columns[int(len(X_d.columns) / 2) :]], drop_first=True, dtype=int
            )
            print('X_p',X_p.shape)
            X = pd.concat([X_a, X_p], axis=1).values
            print('X',X.shape)
            y = df[8].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
        elif args.dataset.dataset_name == 'avazu':
            df = pd.read_csv(DATA_PATH+"avazu/train")
            df = df.sample(frac=0.02, replace=False, random_state=42)
            y = df["click"].values
            feats = process_sparse_feats(df, df.columns[2:])
            xp_idx = df.columns[-8:].tolist()
            xp_idx.insert(0,'C1')
            xa_idx = df.columns[2:-8].tolist()
            xa_idx.remove('C1')
            X_p = feats[xp_idx] # C14-C21
            print('X_p shape',X_p.shape)
            X_a = feats[xa_idx]
            print('X_a shape',X_a.shape)
            X = pd.concat([X_a, X_p], axis=1).values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=False)

        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_dst = (X_train,y_train)
        test_dst = (X_test,y_test)
    elif args.dataset.dataset_name in TEXT_DATA:
        if args.dataset.dataset_name == 'news20':
            texts, labels, labels_index = [], {}, []
            Text_dir = DATA_PATH+'news20/'
            for name in sorted(os.listdir(Text_dir)):
                #  every file_folder under the root_file_folder should be labels with a unique number
                labels[name] = len(labels) # 
                path = join(Text_dir, name)
                for fname in sorted(os.listdir(path)):
                    if fname.isdigit():# The training set we want is all have a digit name
                        fpath = join(path,fname)
                        labels_index.append(labels[name])
                        # skip header
                        f = open(fpath, encoding='latin-1')
                        t = f.read()
                        texts.append(t)
                        f.close()
            #MAX_SEQUENCE_LENGTH = 1000
            #MAX_NB_WORDS = 20000
            #tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
            #tokenizer.fit_on_texts(texts)
            #sequences = tokenizer.texts_to_sequences(texts)
            # word_index = tokenizer.word_index
            # vocab_size = len(word_index) + 1
            #half_dim = int(MAX_SEQUENCE_LENGTH/2) # 500
            #X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            vectorizer = TfidfVectorizer() 
            X = vectorizer.fit_transform(texts)
            X = np.array(X.A)
            y = np.array(labels_index)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.current_seed)
            # ADDED: in config: input_dim = X.shape[1]//2 need to change according to categories included
            half_dim = int(X.shape[1]//2) #42491
        
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        train_dst = (X_train,y_train)
        test_dst = (X_test,y_test)

    else:
        assert args.dataset.dataset_name == 'mnist', "dataset not supported yet"
    
    if len(train_dst) == 2:
        if not args.dataset.dataset_name in GRAPH_DATA:
            if not args.dataset.dataset_name == 'nuswide':
                train_dst = (train_dst[0].to(args.runtime.device),train_dst[1].to(args.runtime.device))
                test_dst = (test_dst[0].to(args.runtime.device),test_dst[1].to(args.runtime.device))
            else:
                train_dst = ([train_dst[0][0].to(args.runtime.device),train_dst[0][1].to(args.runtime.device)],train_dst[1].to(args.runtime.device))
                test_dst = ([test_dst[0][0].to(args.runtime.device),test_dst[0][1].to(args.runtime.device)],test_dst[1].to(args.runtime.device))
            train_dst = dataset_partition(args,index,train_dst,half_dim)
            test_dst = dataset_partition(args,index,test_dst,half_dim)
        else:
            train_dst, args = dataset_partition(args,index,train_dst,half_dim)
            test_dst = ([deepcopy(train_dst[0][0]),deepcopy(train_dst[0][1]),test_dst[0][2]],test_dst[1])
    elif len(train_dst) == 3:
        if not args.dataset.dataset_name in GRAPH_DATA:
            if not args.dataset.dataset_name == 'nuswide':
                train_dst = (train_dst[0].to(args.runtime.device),train_dst[1].to(args.runtime.device),train_dst[2].to(args.runtime.device))
                test_dst = (test_dst[0].to(args.runtime.device),test_dst[1].to(args.runtime.device),test_dst[2].to(args.runtime.device))
            else:
                train_dst = ([train_dst[0][0].to(args.runtime.device),train_dst[0][1].to(args.runtime.device)],train_dst[1].to(args.runtime.device),train_dst[2].to(args.runtime.device))
                test_dst = ([test_dst[0][0].to(args.runtime.device),test_dst[0][1].to(args.runtime.device)],test_dst[1].to(args.runtime.device),test_dst[2].to(args.runtime.device))
            train_dst = dataset_partition(args,index,train_dst,half_dim)
            test_dst = dataset_partition(args,index,test_dst,half_dim)
        else:
            train_dst, args = dataset_partition(args,index,train_dst,half_dim)
            test_dst = ([deepcopy(train_dst[0][0]),deepcopy(train_dst[0][1]),test_dst[0][2]],test_dst[1],test_dst[2])
    # important
    return args, half_dim, train_dst, test_dst
