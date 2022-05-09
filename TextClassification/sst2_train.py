#!/usr/bin/env python
# coding: utf-8

# ### Install Packages

import torch
device='cuda' if torch.cuda.is_available() else 'cpu' 

#
# get_ipython().system('pip install transformers #4.8.2')
# get_ipython().system('pip install datasets')
# get_ipython().system('pip install textbrewer')

import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer,BertConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from datasets import load_dataset,load_metric


# ### Prepare dataset

train_dataset = load_dataset('glue', 'sst2', split='train')
val_dataset = load_dataset('glue', 'sst2', split='validation')
test_dataset = load_dataset('glue', 'sst2', split='test')

train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
val_dataset = val_dataset.remove_columns(['label'])
test_dataset = test_dataset.remove_columns(['label'])
train_dataset = train_dataset.remove_columns(['label'])


model = BertForSequenceClassification.from_pretrained('bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

MAX_LENGTH = 128
train_dataset = train_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
val_dataset = val_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
test_dataset = test_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

out_directory = os.getcwd() + "/results"
log_directory = os.getcwd() + "/logs"
directory = os.getcwd()

print(directory)

#start training 
training_args = TrainingArguments(
    output_dir=out_directory,          
    learning_rate=1e-4,
    num_train_epochs=5,              
    per_device_train_batch_size=32,                
    per_device_eval_batch_size=32,                
    logging_dir=log_directory,            
    logging_steps=100,
    do_train=True,
    do_eval=True,
    no_cuda=False,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
    compute_metrics=compute_metrics
)

train_out = trainer.train()

save_directory = directory + "/sst2_teacher_model.pt"
torch.save(model.state_dict(), save_directory)



out_directory2 = os.getcwd() + "/results2"
log_directory2 = os.getcwd() + "/logs2"
directory = os.getcwd()

print(directory)


model = BertForSequenceClassification.from_pretrained('bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#start training
training_args = TrainingArguments(
    output_dir=out_directory2,
    learning_rate=1e-4,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_dir=log_directory2,
    logging_steps=100,
    do_train=True,
    do_eval=True,
    no_cuda=False,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

train_out = trainer.train()

save_directory2 = directory + "/sst2_teacher_model2.pt"
torch.save(model.state_dict(), save_directory2)



