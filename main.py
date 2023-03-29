
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import torch
import time
import ast
import sys
import os

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, cohen_kappa_score

from ASAG import BertDataBunch, BertLearner,  roc_auc, accuracy_multilabel, cohens_kappa

#train = pd.read_csv('data/LinnData/MIFreq_KI/train2.csv')
#val = pd.read_csv('data/LinnData/MIFreq_KI/val.csv')
#test = pd.read_csv('data/LinnData/MIFreq_KI/test.csv')

# we don't really need the test set, so add to train.
#train = pd.concat([train, test], axis=0)

# best w/ train_longer_4cat, val_4cat, 12, 4, 1e-3

def test_df_to_list(filepath="data/LinnData/MIFreq_KI/test.csv"):
    test_df = pd.read_csv(filepath)
    text_list = [x for x in test_df['text']]
    ground_truth_list = test_df[['1', '2', '3', '4', '5']].values.tolist()
    return text_list, ground_truth_list


def train_bert_linn(batch_size=8, finetune_model=None, epochs=3, lr=3e-3, model_name='MIFreq_KI_temp', save=False,
                    data_path='data/LinnData/MIFreq_KI/',
                    log_path='output/ASAG_models/logs/', output_dir='output/ASAG_models/',
                    text_column='text', labels=None, run_test=False):

    batch_size = int(batch_size)
    epochs = int(epochs)
    lr = float(lr)

    if labels is None:
        labels = ['1', '2', '3', '4']

    DATA_PATH = Path(data_path)
    LOG_PATH = Path(log_path)
    OUTPUT_DIR = Path(output_dir)

    databunch = BertDataBunch(DATA_PATH,
                              tokenizer='bert-large-uncased',
                              train_file='train_plus_test.csv',
                              val_file='val.csv',
                              label_file='labels.csv',
                              label_dir=DATA_PATH,
                              text_col=text_column,
                              label_col=labels,
                              batch_size_per_gpu=batch_size,
                              max_seq_length=512,
                              multi_gpu=False,
                              multi_label=True,
                              model_type='bert')
    # logging
    logger = logging.getLogger()
    log_ext = time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(LOG_PATH, log_ext)

    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    print("LOGGING TO: ", logdir)
    logging.basicConfig(filename='logs.log', level=logging.DEBUG, filemode='w')

    if not finetune_model:
        fine_tune_path = None
    else:
        fine_tune_path = output_dir + finetune_model + '/pytorch_model.bin'

    metrics = [{'name': 'roc_auc', 'function': roc_auc},
               {'name': 'accuracy_multilabel', 'function': accuracy_multilabel},
               # {'name': 'f1_multilabel', 'function': f1},
               {'name': 'cohens_kappa', 'function': cohens_kappa}]

    learner = BertLearner.from_pretrained_model(
        databunch,
        pretrained_path='bert-large-uncased',
        metrics=metrics,
        device=torch.device("cuda"),
        logger=logger,
        output_dir=OUTPUT_DIR,
        finetuned_wgts_path=fine_tune_path,
        grad_accumulation_steps=1,
        warmup_steps=500,
        multi_gpu=False,
        is_fp16=True,
        multi_label=True,
        logging_steps=50)

    learner.fit(epochs=epochs,
                lr=lr,
                validate=False,  # Evaluate the model after each epoch
                schedule_type="warmup_cosine",
                optimizer_type="lamb")

    validation = learner.validate()

    if ast.literal_eval(run_test):
        text_list, test_gt = test_df_to_list(os.path.join(DATA_PATH, 'test.csv'))
        predictions = learner.predict_batch(text_list)
        [x.sort(key=lambda x: x[0]) for x in predictions]

        predictions_formatted = []
        for pred_list in predictions:
            predictions_formatted.append([x[1] for x in pred_list])

        test_gt_acc = [np.argmax(x) for x in test_gt]
        preds_acc = [np.argmax(x) for x in predictions_formatted]

        roc = roc_auc_score(test_gt, predictions_formatted, average='micro')
        acc = accuracy_score(test_gt_acc, preds_acc)
        #f1 = f1_score(test_gt_acc, preds_acc, average='weighted')
        ck = cohen_kappa_score(test_gt_acc, preds_acc, weights='quadratic')

        test_bundle = (roc, acc, ck)
    else:
        test_bundle = None

    if save:
        learner.save_model(path_ext=model_name + '/pytorch_model.bin')
    print(test_bundle, validation)
    return test_bundle, validation


train_bert_linn(batch_size=sys.argv[1], finetune_model=None, epochs=sys.argv[2], lr=sys.argv[3],
                model_name=sys.argv[4], save=True, data_path='data/LinnData/MIFreq_KI/',
                log_path='output/ASAG_models/logs/', output_dir='output/ASAG_models/',
                text_column='text', labels=None, run_test=sys.argv[5])

#main.py 8 3 3e-3 'MIFreq_KI_temp' False

#batch: 16, 6 epochs, lr 6e-3