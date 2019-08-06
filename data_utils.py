import re

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_fscore_support

import matplotlib.pyplot as plt
import re

import numpy as np
import torch

class TensorIndexDataset(TensorDataset):
    def __getitem__(self, index):
        """
        Returns in addition to the actual data item also its index (useful when assign a prediction to a item)
        """
        return index, super().__getitem__(index)
    
def text_to_train_tensors(texts, tokenizer, max_seq_length):
    train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:max_seq_length - 1], texts))
    train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))
    train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=max_seq_length, truncating="post", padding="post",
                                     dtype="int")

    train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]

    # to tensors
    # train_tokens_tensor, train_masks_tensor
    return torch.tensor(train_tokens_ids), torch.tensor(train_masks)


def to_dataloader(texts, extras, ys,
                 tokenizer,
                 max_seq_length,
                 batch_size,
                 dataset_cls=TensorDataset,
                 sampler_cls=RandomSampler):
    """
    Convert raw input into PyTorch dataloader
    """
    #train_y = train_df[labels].values

    # Labels
    train_y_tensor = torch.tensor(ys).float()

    if texts is not None and extras is not None:
        # All features
        train_tokens_tensor, train_masks_tensor = text_to_train_tensors(texts, tokenizer, max_seq_length)
        train_extras_tensor = torch.tensor(extras, dtype=torch.float)

        train_dataset = dataset_cls(train_tokens_tensor, train_masks_tensor, train_extras_tensor, train_y_tensor)
    elif texts is not None and extras is None:
        # Text only
        train_tokens_tensor, train_masks_tensor = text_to_train_tensors(texts, tokenizer, max_seq_length)
        train_dataset = dataset_cls(train_tokens_tensor, train_masks_tensor, train_y_tensor)
    elif texts is None and extras is not None:

        train_extras_tensor = torch.tensor(extras, dtype=torch.float)
        train_dataset = dataset_cls(train_extras_tensor, train_y_tensor)
    else:
        raise ValueError('Either texts or extra must be set.')

    train_sampler = sampler_cls(train_dataset)
    
    return DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)


def get_extras_gender(df, extra_cols, author2vec, author2gender, with_vec=True, with_gender=True, on_off_switch=False):
    """
    Build matrix for extra data (i.e. author embeddings + gender)
    """
    if with_vec:
        AUTHOR_DIM = len(next(iter(author2vec.values())))

        if on_off_switch:
            AUTHOR_DIM += 1  # One additional dimension of binary (1/0) if embedding is available
    else:
        AUTHOR_DIM = 0
        
    if with_gender:
        GENDER_DIM = len(next(iter(author2gender.values())))
    else:
        GENDER_DIM = 0
        
    extras = np.zeros((len(df), len(extra_cols) + AUTHOR_DIM + GENDER_DIM))
    vec_found_selector = [False] * len(df)
    gender_found_selector = [False] * len(df)

    vec_found_count = 0 
    gender_found_count = 0
    
    for i, authors in enumerate(df['authors']):
        # simple extras
        extras[i][:len(extra_cols)] = df[extra_cols].values[i]

        # author vec
        if with_vec:
            for author in authors.split(';'):
                if author in author2vec:
                    if on_off_switch:
                        extras[i][len(extra_cols):len(extra_cols) + AUTHOR_DIM - 1] = author2vec[author]
                        extras[i][len(extra_cols) + AUTHOR_DIM] = 1
                    else:
                        extras[i][len(extra_cols):len(extra_cols)+AUTHOR_DIM] = author2vec[author]

                    vec_found_count += 1
                    vec_found_selector[i] = True
                    break
        
        # author gender
        if with_gender:
            for author in authors.split(';'):
                first_name = author.split(' ')[0]
                if first_name in author2gender:
                    extras[i][len(extra_cols)+AUTHOR_DIM:] = author2gender[first_name]
                    gender_found_count += 1
                    gender_found_selector[i] = True
                    break
                
    return extras, vec_found_count, gender_found_count, vec_found_selector, gender_found_selector


def get_best_thresholds(labels, test_y, outputs, plot=False):
    """
    Hyper parameter search for best classification threshold
    """
    t_max = [0] * len(labels)
    f_max = [0] * len(labels)

    for i, label in enumerate(labels):
        ts = []
        fs = []

        for t in np.linspace(0.1, 0.99, num=50):
            p, r, f, _ = precision_recall_fscore_support(test_y[:,i], np.where(outputs[:,i]>t, 1, 0), average='micro')
            ts.append(t)
            fs.append(f)
            if f > f_max[i]:
                f_max[i] = f
                t_max[i] = t

        if plot:
            print(f'LABEL: {label}')
            print(f'f_max: {f_max[i]}')
            print(f't_max: {t_max[i]}')

            plt.scatter(ts, fs)
            plt.show()
            
    return t_max, f_max


def nn_output_to_submission(first_line, df, outputs, output_ids, t_max, labels, most_popular_label):
    """
    Convert BERT-output into submission format (only a single task)
    """
    no_label = 0

    lines = [first_line]

    for idx in output_ids:
        pred_labels = []

        for i, label in enumerate(labels):   
            if outputs[idx][i] > t_max[i]:
                label = re.sub(r'^([-]+)', '', label)  # remove leading -
                pred_labels.append(label)

        if len(pred_labels) == 0:
            no_label += 1

            # If no label was predicted -> just use most popular
            pred_labels = most_popular_label
        else:
            pred_labels = '\t'.join(pred_labels)

        isbn = df['isbn'].values[idx]

        lines.append(f'{isbn}\t{pred_labels}')
        
    return lines, no_label
