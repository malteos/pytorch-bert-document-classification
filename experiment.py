import os
import pickle

import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch import nn
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm

from data_utils import get_extras_gender, to_dataloader, TensorIndexDataset
from config import MAX_SEQ_LENGTH, HIDDEN_DIM, MLP_DIM, AUTHOR_DIM, GENDER_DIM, TRAIN_BATCH_SIZE, NUM_TRAIN_EPOCHS, \
    default_extra_cols, BERT_MODELS_DIR
from models import ExtraBertMultiClassifier, BertMultiClassifier


class Experiment(object):
    """
    Holds all experiment information
    """
    name = None
    output_dir = None
    epochs = None
    batch_size = None
    device = None
    labels = None

    def __init__(self, task, bert_model, classifier_model=None, with_text=True, with_author_gender=True,
                 with_manual=True, with_author_vec=True, author_vec_switch=False, mlp_dim=None):
        self.task = task
        self.bert_model = bert_model
        self.with_text = with_text
        self.with_author_gender = with_author_gender
        self.with_manual = with_manual
        self.with_author_vec = with_author_vec
        self.author_vec_switch = author_vec_switch
        self.classifier_model = classifier_model

        self.mlp_dim = mlp_dim if mlp_dim is not None else MLP_DIM

    def init(self, cuda_device, epochs, batch_size, continue_training):
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        if not torch.cuda.is_available():
            print('CUDA GPU is not available')
            exit(1)

        self.epochs = epochs if epochs is not None else NUM_TRAIN_EPOCHS
        self.batch_size = batch_size if batch_size is not None else TRAIN_BATCH_SIZE

        if not continue_training and os.path.exists(self.get_output_dir()):
            print(f'Output directory exist already: {self.get_output_dir()}')
            exit(1)
        else:
            os.makedirs(self.get_output_dir())

    def get_output_dir(self):
        return os.path.join(self.output_dir, self.name)

    def get_bert_model_path(self):
        return os.path.join(BERT_MODELS_DIR, self.bert_model)

    def get_author_dim(self):
        # Use author switch?
        if self.author_vec_switch:
            author_dim = AUTHOR_DIM + 1
        else:
            author_dim = AUTHOR_DIM

        return author_dim

    def get_extra_cols(self):
        if self.with_manual:
            extra_cols = default_extra_cols
        else:
            extra_cols = []
        return extra_cols

    def prepare_data_loaders(self, df_train_path, df_val_path, extras_dir, test_set=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.with_text:
            tokenizer = BertTokenizer.from_pretrained(self.get_bert_model_path(), do_lower_case=False)
        else:
            tokenizer = None

        # Load external data
        if self.with_author_vec:
            with open(os.path.join(extras_dir, 'author2embedding.pickle'), 'rb') as f:
                author2vec = pickle.load(f)

            print(f'Embeddings avaiable for {len(author2vec)} authors')
        else:
            author2vec = None

        if self.with_author_gender:
            gender_df = pd.read_csv(os.path.join(extras_dir, 'name_gender.csv'))
            author2gender = {
                row['name']: np.array([row['probability'], 0] if row['gender'] == 'M' else [0, row['probability']]) for
                idx, row in gender_df.iterrows()}

            print(f'Gender data avaiable for {len(author2gender)} authors')
        else:
            author2gender = None

        # Load training data
        with open(df_train_path, 'rb') as f:
            train_df, doc_cols, task_b_labels, task_a_labels = pickle.load(f)

        # Define labels (depends on task)
        if self.task == 'a':
            self.labels = task_a_labels
        elif self.task == 'b':
            self.labels = task_b_labels
        else:
            raise ValueError('Invalid task specified')

        if self.with_manual or self.with_author_gender or self.with_author_vec:
            train_extras, vec_found_count, gender_found_count, _, _ = get_extras_gender(
                train_df,
                self.get_extra_cols(),
                author2vec,
                author2gender,
                with_vec=self.with_author_vec,
                with_gender=self.with_author_gender,
                on_off_switch=self.author_vec_switch
            )
        else:
            train_extras = None

        if self.with_text:
            train_texts = [t + '.\n' + train_df['text'].values[i] for i, t in enumerate(train_df['title'].values)]
        else:
            train_texts = None

        train_y = train_df[self.labels].values

        train_dataloader = to_dataloader(train_texts, train_extras, train_y,
                                         tokenizer,
                                         MAX_SEQ_LENGTH,
                                         self.batch_size,
                                         dataset_cls=TensorDataset,
                                         sampler_cls=RandomSampler)

        # Load validation data
        with open(df_val_path, 'rb') as f:
            val_df, _, _, _ = pickle.load(f)

        if self.with_manual or self.with_author_gender or self.with_author_vec:
            val_extras, vec_found_count, gender_found_count, vec_found_selector, gender_found_selector = get_extras_gender(
                val_df,
                self.get_extra_cols(),
                author2vec,
                author2gender,
                with_vec=self.with_author_vec,
                with_gender=self.with_author_gender,
                on_off_switch=self.author_vec_switch,
            )
        else:
            val_extras = None
            vec_found_selector = None

        if self.with_text:
            val_texts = [t + '.\n' + val_df['text'].values[i] for i, t in enumerate(val_df['title'].values)]
        else:
            val_texts = None

        # Is test set?
        # np.zeros((len(test_texts), len(labels)))
        if test_set:
            val_y = np.zeros((len(val_texts), len(self.labels)))
        else:
            val_y = val_df[self.labels].values

        val_dataloader = to_dataloader(val_texts, val_extras, val_y,
                                       tokenizer,
                                       MAX_SEQ_LENGTH,
                                       self.batch_size,
                                       dataset_cls=TensorIndexDataset,
                                       sampler_cls=SequentialSampler)

        return train_dataloader, val_dataloader, vec_found_selector, val_df, val_y

    def get_model(self):
        if self.classifier_model is None:
            # No pre-defined model

            extras_dim = len(self.get_extra_cols())
            if self.with_author_vec:
                extras_dim += self.get_author_dim()

            if self.with_author_gender:
                extras_dim += GENDER_DIM

            if extras_dim > 0:
                model = ExtraBertMultiClassifier(
                    bert_model_path=self.get_bert_model_path(),
                    labels_count=len(self.labels),
                    hidden_dim=HIDDEN_DIM,
                    extras_dim=extras_dim,
                    mlp_dim=self.mlp_dim,
                )
            else:
                # Text only: Standard BERT classifier
                model = BertMultiClassifier(
                    bert_model_path=self.get_bert_model_path(),
                    labels_count=len(self.labels),
                    hidden_dim=HIDDEN_DIM,
                )
        else:
            model = self.classifier_model

        return model

    def train(self, model, optimizer, train_dataloader):
        for epoch_num in range(self.epochs):
            model.train()
            train_loss = 0

            print(f'Epoch: {epoch_num + 1}/{self.epochs}')

            # for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration")):
            for step_num, batch_data in enumerate(tqdm(train_dataloader, desc="Iteration")):

                if self.with_text and (
                        self.with_manual or self.with_author_gender or self.with_author_vec):
                    # Full features
                    token_ids, masks, extras, gold_labels = tuple(t.to(self.device) for t in batch_data)
                    probas = model(token_ids, masks, extras)
                elif self.with_text:
                    # Text only
                    token_ids, masks, gold_labels = tuple(t.to(self.device) for t in batch_data)
                    probas = model(token_ids, masks)
                else:
                    # Extras only
                    extras, gold_labels = tuple(t.to(self.device) for t in batch_data)
                    probas = model(extras)

                loss_func = nn.BCELoss()
                batch_loss = loss_func(probas, gold_labels)
                train_loss += batch_loss.item()

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # clear_output(wait=True)

            print(f'\r{epoch_num} loss: {train_loss / (step_num + 1)}')

            print(str(torch.cuda.memory_allocated(self.device) / 1000000) + 'M')

        return model

    def eval(self, model, data_loader):

        # Validation
        model.eval()

        output_ids = []
        outputs = None

        with torch.no_grad():
            for step_num, batch_item in enumerate(data_loader):
                batch_ids, batch_data = batch_item

                if self.with_text and (
                        self.with_manual or self.with_author_gender or self.with_author_vec):
                    # Full features
                    token_ids, masks, extras, _ = tuple(t.to(self.device) for t in batch_data)
                    logits = model(token_ids, masks, extras)
                elif self.with_text:
                    # Text only
                    token_ids, masks, _ = tuple(t.to(self.device) for t in batch_data)
                    logits = model(token_ids, masks)
                else:
                    # Extras only
                    extras, _ = tuple(t.to(self.device) for t in batch_data)
                    logits = model(extras)

                numpy_logits = logits.cpu().detach().numpy()

                if outputs is None:
                    outputs = numpy_logits
                else:
                    outputs = np.vstack((outputs, numpy_logits))

                output_ids += batch_ids.tolist()

        print(f'Evaluation completed for {len(outputs)} items')

        return output_ids, outputs
