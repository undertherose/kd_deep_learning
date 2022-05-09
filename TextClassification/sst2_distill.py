#!/usr/bin/env python
# coding: utf-8

# ### Install Packages



import torch
device='cuda' if torch.cuda.is_available() else 'cpu' 

# get_ipython().system('pip install transformers #4.8.2')
# get_ipython().system('pip install datasets')
# get_ipython().system('pip install textbrewer')


import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer,BertConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from datasets import load_dataset,load_metric

directory = os.getcwd()
save_directory = directory + "/sst2_teacher_model.pt"
save_directory2 = directory + "/sst2_teacher_model2.pt"
bert_dir = directory + '/student_config/bert_base_cased_config/bert_config.json'
bert_dir_T3 = directory + "/student_config/bert_base_cased_config/bert_config_L3.json"


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


# ### Distillation - 1, General Distiller


from torch.utils.data import DataLoader, RandomSampler
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)



import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from transformers import BertForSequenceClassification, BertConfig, AdamW,BertTokenizer
from transformers import get_linear_schedule_with_warmup


bert_config_T3 = BertConfig.from_json_file(bert_dir_T3)
bert_config_T3.output_hidden_states = True

student_model = BertForSequenceClassification(bert_config_T3)
student_model.to(device=device)


bert_config = BertConfig.from_json_file(bert_dir)
bert_config.output_hidden_states = True
teacher_model = BertForSequenceClassification(bert_config)
teacher_model.load_state_dict(torch.load(save_directory))
teacher_model.to(device=device)

num_epochs = 20
num_training_steps = len(train_dataloader) * num_epochs
optimizer = AdamW(student_model.parameters(), lr=1e-4)

scheduler_class = get_linear_schedule_with_warmup
scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}


def simple_adaptor(batch, model_outputs):
    return {'logits': model_outputs.logits, 
        'hidden': model_outputs.hidden_states, 
        'attention': model_outputs.attentions}

distill_config = DistillationConfig(
    intermediate_matches=[    
     {'layer_T':0, 'layer_S':0, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
     {'layer_T':8, 'layer_S':2, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1}])
train_config = TrainingConfig()

distiller = GeneralDistiller(
    train_config=train_config, distill_config=distill_config,
    model_T=teacher_model, model_S=student_model, 
    adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)

tic = time.time()
print(tic)
with distiller:
    distiller.train(optimizer, train_dataloader, num_epochs, scheduler_class=scheduler_class, scheduler_args = scheduler_args, callback=None)
tac = time.time()


trainingtime = tac-tic

print(trainingtime)




tic = time.time()



test_model = BertForSequenceClassification(bert_config_T3)
model_dir = directory + "/saved_models/gs4210.pkl"
test_model.load_state_dict(torch.load(model_dir))



from torch.utils.data import DataLoader
eval_dataloader = DataLoader(val_dataset, batch_size=8)


metric= load_metric("accuracy")
test_model.eval()
for batch in eval_dataloader:
    batch = {k: v for k, v in batch.items()}
    with torch.no_grad():
        outputs = test_model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
print("General Acc Val")
print(metric.compute())




test_model = BertForSequenceClassification(bert_config_T3)
model_dir = directory + "/saved_models/gs4210.pkl"
test_model.load_state_dict(torch.load(model_dir))#gs4210 is the distilled model weights file


from torch.utils.data import DataLoader
eval_dataloader = DataLoader(train_dataset, batch_size=8)



metric= load_metric("accuracy")
test_model.eval()
for batch in eval_dataloader:
    batch = {k: v for k, v in batch.items()}
    with torch.no_grad():
        outputs = test_model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print("General Acc Train")
print(metric.compute())


# ### Distillation 2: Basic Distiller


bert_config_T3 = BertConfig.from_json_file(bert_dir_T3)
bert_config_T3.output_hidden_states = True

student_model = BertForSequenceClassification(bert_config_T3)
student_model.to(device=device)

bert_config = BertConfig.from_json_file(bert_dir)
bert_config.output_hidden_states = True
teacher_model = BertForSequenceClassification(bert_config)
teacher_model.load_state_dict(torch.load(save_directory))
teacher_model.to(device=device)


#Basic Distiller
from textbrewer import BasicDistiller

num_epochs = 20
num_training_steps = len(train_dataloader) * num_epochs
optimizer = AdamW(student_model.parameters(), lr=1e-4)

scheduler_class = get_linear_schedule_with_warmup
scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}


def simple_adaptor(batch, model_outputs):
    return {'logits': model_outputs.logits, 
        'hidden': model_outputs.hidden_states, 
        'attention': model_outputs.attentions}

distill_config = DistillationConfig()
train_config = TrainingConfig()

distiller = BasicDistiller(
    train_config=train_config, distill_config=distill_config,
    model_T=teacher_model, model_S=student_model, 
    adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)

tic = time.time()
print(tic)
with distiller:
    distiller.train(optimizer, train_dataloader, num_epochs, scheduler_class=scheduler_class, scheduler_args = scheduler_args, callback=None)
tac = time.time()


trainingtime = tac-tic

print(trainingtime)

test_model = BertForSequenceClassification(bert_config_T3)
model_dir = directory + "/saved_models/gs4210.pkl"
test_model.load_state_dict(torch.load(model_dir))

from torch.utils.data import DataLoader
eval_dataloader = DataLoader(val_dataset, batch_size=8)

metric= load_metric("accuracy")
test_model.eval()
for batch in eval_dataloader:
    batch = {k: v for k, v in batch.items()}
    with torch.no_grad():
        outputs = test_model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
print("Basic Acc Val")
print(metric.compute())

test_model = BertForSequenceClassification(bert_config_T3)
model_dir = directory + "/saved_models/gs4210.pkl"
test_model.load_state_dict(torch.load(model_dir))

from torch.utils.data import DataLoader
eval_dataloader = DataLoader(train_dataset, batch_size=8)

metric= load_metric("accuracy")
test_model.eval()
for batch in eval_dataloader:
    batch = {k: v for k, v in batch.items()}
    with torch.no_grad():
        outputs = test_model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print("Basic Acc Train")
print(metric.compute())


# ### Distillation 3: Multi Teacher Distiller

bert_config_T3 = BertConfig.from_json_file(bert_dir_T3)
bert_config_T3.output_hidden_states = True

student_model = BertForSequenceClassification(bert_config_T3)
student_model.to(device=device)


bert_config = BertConfig.from_json_file(bert_dir)
bert_config.output_hidden_states = True
teacher_model = BertForSequenceClassification(bert_config)
teacher_model.load_state_dict(torch.load(save_directory))
teacher_model.to(device=device)

teacher_model2 = BertForSequenceClassification(bert_config)
teacher_model2.load_state_dict(torch.load(save_directory2))
teacher_model2.to(device=device)

model_Ts = []
model_Ts.append(teacher_model)
model_Ts.append(teacher_model2)


# In[ ]:


#Multiteacher Distiller
import time
from textbrewer import MultiTeacherDistiller
# from modeling import BertForGLUESimple, BertForGLUESimpleAdaptorTrain, BertForGLUESimpleAdaptor

num_epochs = 20
num_training_steps = len(train_dataloader) * num_epochs

optimizer = AdamW(student_model.parameters(), lr=1e-4)

scheduler_class = get_linear_schedule_with_warmup

scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}


def simple_adaptor(batch, model_outputs):
#     print(model_outputs)
    return {'logits': model_outputs.logits, 
        'hidden': model_outputs.hidden_states, 
        'attention': model_outputs.attentions}

distill_config = DistillationConfig()
train_config = TrainingConfig()

distiller = MultiTeacherDistiller(
    train_config=train_config, distill_config=distill_config,
    model_T=model_Ts, model_S=student_model, 
    adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)

tic = time.time()
print(tic)
with distiller:
    distiller.train(optimizer, train_dataloader, num_epochs, scheduler_class=scheduler_class, scheduler_args = scheduler_args, callback=None)
tac = time.time()


trainingtime = tac-tic

print(trainingtime)

test_model = BertForSequenceClassification(bert_config_T3)
model_dir = directory + "/saved_models/gs4210.pkl"
test_model.load_state_dict(torch.load(model_dir))

from torch.utils.data import DataLoader
eval_dataloader = DataLoader(val_dataset, batch_size=8)

metric= load_metric("accuracy")
test_model3.eval()
for batch in eval_dataloader:
    batch = {k: v for k, v in batch.items()}
    with torch.no_grad():
        outputs = test_model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print("Multi Acc Val")
print(metric.compute())

test_model = BertForSequenceClassification(bert_config_T3)
model_dir = directory + "/saved_models/gs4210.pkl"
test_model.load_state_dict(torch.load(model_dir))

from torch.utils.data import DataLoader
eval_dataloader = DataLoader(train_dataset, batch_size=8)

metric= load_metric("accuracy")
test_model3.eval()
for batch in eval_dataloader:
    batch = {k: v for k, v in batch.items()}
    with torch.no_grad():
        outputs = test_model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print("Multi Acc Train")
print(metric.compute())

