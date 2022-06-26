#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch
import torch.nn as nn
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import re
import string
import math
from collections import defaultdict
import random
import warnings
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc,plot_roc_curve,accuracy_score,auc,accuracy_score,f1_score

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig,TrainingArguments,Trainer
from datasets import Dataset,DatasetDict
import torch.nn.utils.prune as prune
import wandb
import optuna
from optimum.onnxruntime import ORTConfig, ORTQuantizer
from prettytable import PrettyTable
import copy


warnings.filterwarnings("ignore", message="No positive samples in y_true, true positive value should be meaningless")
warnings.filterwarnings("ignore", message="No negative samples in y_true, false positive value should be meaningless")


# In[2]:


# Setting a random seed so we can compare the models
random.seed(0)


# In[4]:


## Activating cuda

get_ipython().system('Conda activate tf_GPU')


# In[5]:


## Checking if cuda is available

torch.cuda.is_available()


# In[6]:


## Getting the torch device name

torch.cuda.get_device_name(0)


# In[3]:


## Defining the models we are going to use

model_name_XLNet='xlnet-base-cased'
model_name_Roberta='roberta-base'


# # Defining Class Settings

# In[4]:


class PreTrainedModel():
    
    # Initializing the model's attributes
    
    def __init__(self,model_name,lr,loss_func,dataset):
        self.model_name=model_name
        self.lr=lr
        self.model=AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2,return_dict=True)
        self.optimizer=torch.optim.Adam(self.model.parameters(), self.lr)
        self.loss_func = loss_func
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.dataset=dataset
        self.tokenized_dataset=None
    
    # A function to tokenize reviews
    
    def tokenize(self,parameters):
        self.tokenized_dataset=self.dataset.map(self.tokenizer, input_columns='text', fn_kwargs=parameters)
        for split in self.tokenized_dataset:
                self.tokenized_dataset[split] = self.tokenized_dataset[split].add_column('labels',self.dataset[split]['labels'])
                self.tokenized_dataset[split]=self.tokenized_dataset[split].remove_columns(['text'])
        self.tokenized_dataset.set_format('torch')
        return
    
    # A function to evaluate accuracy and F1 metrics
    
    def metric_fn(self,predictions):
        preds = predictions.predictions.argmax(axis=1)
        labels = predictions.label_ids
        return {'f1': f1_score(labels, preds, average='binary')
               ,'accuracy':accuracy_score(labels,preds)}
    
    # A function to train the model
    
    def train(self,train_args):
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['test'],
            compute_metrics=self.metric_fn)
        
        trainer.train()
    
    # A function to return the model init attribute
    
    def model_init(self):
        return self.model
    
    # A function to perform a hyperparameter random search using XLNet
    
    def hpm_search_XLNet(self,train_args):
        wandb.init(entity="group2tau")
        get_ipython().run_line_magic('env', 'WANDB_PROJECT=XLNet')
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['test'],
            model_init=self.model_init,
            compute_metrics=self.metric_fn)
        
        trainer.hyperparameter_search(direction="maximize", hp_space=self.my_hp_space,n_trials=20)
    
    # A function to perform a hyperparameter random search using XLNet
    
    def hpm_search_Roberta(self,train_args):
        wandb.init(entity="group2tau")
        get_ipython().run_line_magic('env', 'WANDB_PROJECT=Roberta')
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['test'],
            model_init=self.model_init,
            compute_metrics=self.metric_fn)
        
        trainer.hyperparameter_search(direction="maximize", hp_space=self.my_hp_space,n_trials=20)
    
    # A function that returns the hyperparameter search space arguments
    
    def my_hp_space(self,trial):
        return {"learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 3,3),
                "seed": trial.suggest_int("seed", 0, 0),
                "per_device_train_batch_size": trial.suggest_int("per_device_train_batch_size", 16,27),
                "gradient_accumulation_steps":trial.suggest_int("gradient_accumulation_steps",1,6),
                "warmup_steps":trial.suggest_int("warmup_steps",0,500),
                "weight_decay":trial.suggest_float("weight_decay",1e-4,1e-2),
                "per_device_eval_batch_size":trial.suggest_int("per_device_eval_batch_size",16,16)}
    
    # A function to show the modules and parameters in the model
    
    def show_parameters(self):
        table = PrettyTable(["Modules", "Parameters","Sum of Tensor"])
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: 
                continue
            params = parameter.numel()
            total=float(str(parameter.sum()).split(',')[0][7:])
            table.add_row([name, params,total])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
    
    # A function to prune the weights of the feed-forward layer in XLNet with a hyperparameter of the amount to prune
    
    def pruning(self,amount):
        for module,layer in list(self.model.named_modules()):
            if isinstance(layer,torch.nn.modules.linear.Linear) and 'ff' in module:
                print(f'\n{module}:\nold_total_weights = {layer.weight.sum()}')
                m=prune.l1_unstructured(layer,name="weight", amount=amount)
                m=prune.remove(m,name="weight")
                print(f'new_total_weights = {layer.weight.sum()}')
    
    # A function to quantize the weights of the feed-forward layer in XLNet with a hyperparameter of the scale
    
    def quantization(self,scale):
        for i,(module,layer) in enumerate(list(self.model.named_modules())):
            if isinstance(layer,torch.nn.modules.linear.Linear) and 'ff' in module:
                weights=torch.quantize_per_tensor(layer.weight, 0.1, 0, torch.quint8).int_repr()
                list(self.model.named_modules())[i][1].weight.data=weights

    # A function to print the model's size
    
    def print_model_size(self):
        torch.save(self.model.state_dict(), "tmp.pt")
        print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
        os.remove('tmp.pt')


# # Reading the data

# In[5]:


vocab_path='imdb.vocab'

def make_vocab(path):
    # Reading imdb vocab file
    with open(path,encoding="utf8") as file:
        words_list=file.readlines()
    
    # Splitting /n and "" from the words
    words_list=[s.split('\n')[0].split('""')[0] for s in words_list]
    # Generating a list of lists containing the vocab word
    words_list=[[word] for word in words_list]
    return words_list

# A list to hold the imdb vocab
words_list=make_vocab(vocab_path)


# # Exploration and Pre-Processing

# In[6]:


# Function to read the  imdb files

def read_files(path):
    """
    Run this script from the root directory
    """
    for file in glob.glob(f"{path}\*.txt"):
        yield file
        
# Function to clean a review query (lower case, special characters)     

def clean_query(query):
    for punc in string.punctuation:
        if punc == "'":
            query=query.replace(punc,'')
        else:
            query=query.replace(punc,' ')
    return lower(query.replace('br',' '))

# Function to load reviews

def load_reviews(path):
    reviews = {}
    for file_name in read_files(path):
        split=file_name.split('_')
        if len(split)==2:
            review_ind=int(split[0].split('\\')[-1])
            if review_ind not in reviews:
                with open(file_name, encoding='UTF-8') as file:
                    reviews[review_ind]=([clean_query(file.read())],int(split[1].split('.')[0]))
    return reviews

    
# Function to initiate loading of positive reviews

def load_positive_reviews():
    path = "train/pos"
    return load_reviews(path)

# Function to initiate loading of negative reviews

def load_negative_reviews():
    path = "train/neg"
    return load_reviews(path)
    
# Function that creates a word counter in all of the reviews and deletes word with less than 5 appearances

def create_word_counter(reviews):
    reviews_list =reviews
    word_counter = {}
    for review in reviews_list.values():
        review_text = review[0][0]
        for word in review_text.split():
            if word.lower() in word_counter:
                word_counter[word.lower()] +=1
            else:
                word_counter[word.lower()] =1
                
    # Remove words with less than 5 appearances
    
    words_to_remove = set()
    clean_word_counter = {}
    for word, count in word_counter.items():
        if count < 5: 
            words_to_remove.add(word)
        else:
            clean_word_counter[word] = count

    for review in reviews_list.keys(): 
        review_text = reviews_list[review][0][0]
        text_without_uncommon_words = []
        for word in review_text.split():
            if word not in words_to_remove:
                text_without_uncommon_words.append(word)
        reviews_list[review]=([" ".join(text_without_uncommon_words)],reviews_list[review][1])

        
    return clean_word_counter, reviews_list

### Function that creates a word counter of the top K words that appear
### both in the positive and the negative reviews and deletes them from all reviews 

def top_k_word_counter(all_reviews, k):
    word_counter = {}
    for r in all_reviews:
        reviews_list = r
        for review in reviews_list.values():
            review_text = review[0][0]
            for word in review_text.split():
                if word.lower() in word_counter:
                    word_counter[word.lower()] +=1
                else:
                    word_counter[word.lower()] =1
    return dict(sorted([(k,v) for k,v in word_counter.items()], key=lambda x: x[1], reverse=True)[:k])

def get_word_probs(all_reviews) -> dict:
    """
    Count the number of words in all the documents
    :return: dictionary of word: probability
    """
    log_factor_to_divide = 30
    word_counter = defaultdict(int)
    for r in all_reviews:
        reviews_list = r
        for review in reviews_list.values():
            review_text = review[0][0]
            for word in review_text.split():
                word_counter[word.lower()] += 1
    word_to_prob = sorted([[k,v] for k,v in word_counter.items()], key= lambda x: x[1], reverse=True)
    
    return dict(word_to_prob)

def delete_words_by(all_reviews, min_count=5, top_k=0, probs=False):
    """
    Delete words by critiria
    You can delete by min count of words (default 5 words)
    You can delete by top K of most common words
    You can delete from certain probabaiities of word occurence

    :param all_reviews: list of reviws dictionary
    :param min_count: min count of word count in reviews defaults to 5
    :param top_k: top K most comon words to delete, defaults to 0
    :param probs: delete by word probabilities, defaults to False
    :return: reviews dictionary
    """
    word_counter = {}
    for r in all_reviews:
        reviews_list = r
        for review in reviews_list.values():
            review_text = review[0][0]
            for word in review_text.split():
                if word.lower() in word_counter:
                    word_counter[word.lower()] +=1
    
    words_to_remove = set() # Set of words to remove by criteria
    if top_k > 0:
        words = top_k_word_counter(all_reviews, top_k) # get top k words
        for w in words.keys():
            words_to_remove.add(w) # add top K words to the set of words we will remove


    if min_count > 0: 
        for word, count in word_counter.items():
            if count < min_count: 
                words_to_remove.add(word)
    if probs:
        word_probs = get_word_probs(all_reviews)  
    
    all_reviews_to_return={}
    count=0
    for reviews_dict in all_reviews:
        num_of_reviews=len(reviews_dict.keys())
        for review in reviews_dict.keys(): 
            review_text = reviews_dict[review][0][0]
            text_without_uncommon_words = []
            for word in review_text.split():
                if word not in words_to_remove:
                    if probs:
                        word_deletion_chance = np.random.uniform(0,1)
                        if word_deletion_chance < word_probs[word.lower()]: # Add the word only if it's not deleted
                            text_without_uncommon_words.append(word)
                    
                    else:
                        text_without_uncommon_words.append(word)
            all_reviews_to_return[review+count]=([" ".join(text_without_uncommon_words)],reviews_dict[review][1])
        count+=num_of_reviews
    return all_reviews_to_return
    

# Function to transform the reviews to lower case

def lower(query):
    return " ".join(list(map(lambda x:x.lower(),query.split())))

# Function to concatenate all reviews into one list

def concatsentences(lst,query):
    lst.append(lower(query[0]).split())
    return


def sort_dict(x: dict, reverse=True) -> dict:
    """
    Sort a dictionary to display it on a graph

    :param x: dictionary with some values
    :return: sorted dictionary
    """
    return {k:v for k,v in sorted(x.items(), key=lambda i: i[1], reverse=reverse)}


# ## Exploration 
# We'll explore the data by showing statistics, data types, and Number of unique words, distribution
# 
# In addition, we'll plot some visualization to better understend the data

# In[7]:


def plot_common_words(hist: dict, title, n=20) -> None:
    """
    Plot common word histogram
    :param hist: dictionary with data to plot
    """
    _, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_ylabel('Number of appearances')
    ax.set_xlabel('Word')
    ax.set_xticklabels(labels=list(hist.keys())[:n] ,rotation = 90)
    ax.bar(list(hist.keys())[:n], list(hist.values())[:n])
    plt.show()


# In[8]:


revs = [load_positive_reviews(), load_negative_reviews()] # which reviews you want to take care of - pos, neg or both

all_words_preprocess = delete_words_by(revs)


# Plotting Top Common Words before dropping common words

plot_common_words(sort_dict(create_word_counter(all_words_preprocess)[0]),'Most common 20 words - Before preproccessing')
plot_common_words(sort_dict(create_word_counter(all_words_preprocess)[0]),'Top 10 common words',10)


# In[9]:


## Calculating the average review length and average review rating before cleaning the reviews

all_reviews_uncleaned=delete_words_by(revs, 0, 0, False)

num_of_reviews=0
len_of_reviews=0
rating=0

for key,value in all_reviews_uncleaned.items():
    len_of_reviews+=len(value[0][0])
    rating+=value[1]
    num_of_reviews+=1

avg_review_length_cleaned=round(len_of_reviews/num_of_reviews,0)
avg_review_rating=round(rating/num_of_reviews,2)

print(f"Average review length before pre-processing is : {avg_review_length_cleaned}\n")
print(f"Average review rating is : {avg_review_rating}")


# In[ ]:





# In[10]:


# Dropping top 5 common words

all_words_top_k_dropped = delete_words_by(revs,min_count=10,top_k=5)
plot_common_words(sort_dict(create_word_counter(all_words_top_k_dropped)[0]),'Most common 10 words - Top 5 dropped',10)


# In[11]:




all_reviews_cleaned=delete_words_by(revs, 10, 5, True)
clean_counter= sort_dict(create_word_counter(all_reviews_cleaned)[0])

# After dropping top 5 common words + probability drop

plot_common_words(clean_counter, 'Most common 10 words - Top 5 dropped + probability drop',10)


# In[12]:


## Calculating the average review length after cleaning the reviews

num_of_reviews=0
len_of_reviews=0

for key,value in all_reviews_cleaned.items():
    len_of_reviews+=len(value[0][0])
    rating+=value[1]
    num_of_reviews+=1

avg_review_length_cleaned=round(len_of_reviews/num_of_reviews,0)

print(f"Average review length after pre-processing is : {avg_review_length_cleaned}")


# ## Pre-Processing

# In[11]:


# Loading all reviews into positive and negative lists

sentences=[]
revs = [load_positive_reviews(),load_negative_reviews()] # which reviews you want to take care of - pos, neg or both


all_reviews_cleaned=delete_words_by(revs, 10, 5, True)


# Splitting the unified reviews into negative and positive by the rating (higher or lower than 5)

pos_reviews={}
neg_reviews={}
num_of_reviews=int(len(all_reviews_cleaned)/2)
for key,value in all_reviews_cleaned.items():
    if value[1]>=5:
        pos_reviews[key]=value
    else:
        neg_reviews[key-num_of_reviews]=value


for query in pos_reviews.keys():
    concatsentences(sentences,pos_reviews[query][0])

for query in neg_reviews.keys():
    concatsentences(sentences,neg_reviews[query][0])


# In[12]:


neg_reviews


# In[13]:


sentences


# # Arranging the data

# In[14]:


## Shuffling concatenated data to keep the order of the tagging

def shuffledata(x_lst,y_lst):
    z=list(zip(x_lst,y_lst))
    random.shuffle(z)
    x,y=zip(*z)
    return x,y

# A function to split the data into text sets and labels sets

def split_data(test_sz):
    # Loading positive and negative reviews
    x_pos=[pos_reviews[i][0] for i in range(len(pos_reviews.keys()))]
    y_pos=[1 for i in range(len(pos_reviews.keys()))]
    x_neg=[neg_reviews[i][0] for i in range(len(neg_reviews.keys()))]
    y_neg=[0 for i in range(len(neg_reviews.keys()))]


    ## Stratifying the train and test sets
    
    x_pos_train, x_pos_test, y_pos_train, y_pos_test = train_test_split(x_pos, y_pos, test_size=test_sz, random_state=42)
    x_neg_train, x_neg_test, y_neg_train, y_neg_test = train_test_split(x_neg, y_neg, test_size=test_sz, random_state=42)

    x_combined_train=x_pos_train+x_neg_train
    x_combined_test=x_pos_test+x_neg_test
    y_combined_train=y_pos_train+y_neg_train
    y_combined_test=y_pos_test+y_neg_test

    x_final_train,y_final_train=shuffledata(x_combined_train,y_combined_train)
    x_final_test,y_final_test=shuffledata(x_combined_test,y_combined_test)
    
    return x_final_train,y_final_train,x_final_test,y_final_test

# Splitting the data into train and validation sets (75% train, 25% validation)

x_final_train,y_final_train,x_final_test,y_final_test=split_data(test_sz=0.25)


# In[15]:


# Saving the train and validation sets in dataframes (so we can use the Datasets object)

train_df=pd.DataFrame()
test_df=pd.DataFrame()

train_df['text']=[text[0] for text in x_final_train]
train_df['labels']=y_final_train

test_df['text']=[text[0] for text in x_final_test]
test_df['labels']=y_final_test


# In[16]:


train_df


# In[17]:


# Generating a datasets object so we can run the models using GPU (allows us to run on a batch size higher than 1)

train_ds=Dataset.from_pandas(train_df[:])
test_ds=Dataset.from_pandas(test_df[:])



dataset_united=DatasetDict({'train':train_ds,"test":test_ds})
dataset_united


# # Training Process

# ## Hyperparameter Random Search - XLNet

# In[448]:


# Defining the loss function and loading the XLNet pretrained model so we can run a hyperparameter search

loss_func=torch.nn.BCELoss()

model1XLNet=PreTrainedModel(model_name_XLNet,1e-5,loss_func,dataset_united)


# In[23]:


# Tokenizing the words in the reviews using XLNet loaded model

model1XLNet.tokenize({"max_length": 200, "truncation": True, "padding": "max_length"})


# In[24]:


# Defining saving path, and initializing training arguments for hyperparameter search of XLNet and RoBerta

OUT_PATH=r"C:\Users\user\Desktop\Group2"

args = TrainingArguments(output_dir=OUT_PATH,
                         overwrite_output_dir=True,
                         save_strategy='no',
                         greater_is_better=True,
                         evaluation_strategy='epoch',
                         do_train=True,
                         logging_strategy='epoch',
                         report_to='wandb')


# In[25]:


### Running a hyperparameter search on XLNet

model1XLNet.hpm_search_XLNet(args)


# In[ ]:





# ## Hyperparameter Random Search - RoBERTa

# In[27]:


# Loading the Roberta pretrained model so we can run a hyperparameter search

model2Roberta=PreTrainedModel(model_name_Roberta,1e-5,loss_func,dataset_united)


# In[28]:


# Tokenizing the words in the reviews using RoBerta loaded model

model2Roberta.tokenize({"max_length": 200, "truncation": True, "padding": "max_length"})


# In[29]:


### Running a hyperparameter search on RoBerta

model2Roberta.hpm_search_Roberta(args)


# # Training on the hyperparameters found

# ## Training XLNet

# In[18]:


# Creating the reporting enviornment and training arguments for XLNet final model to train

wandb.init(project="FinalModel1",entity="group2tau")
OUT_PATH=r"C:\Users\user\Desktop\Group2"
args = TrainingArguments(output_dir=OUT_PATH,
                         overwrite_output_dir=True,
                         per_device_train_batch_size=17,
                         per_device_eval_batch_size=16,
                         learning_rate=2.6785e-5,
                         weight_decay=9.08e-3,
                         save_strategy='no',
                         greater_is_better=True,
                         evaluation_strategy='epoch',
                         do_train=True,
                         num_train_epochs=3,
                         gradient_accumulation_steps=6,
                         logging_strategy='epoch',
                         warmup_steps=426,
                         report_to='wandb')


# In[19]:


# Configuring loss function and loading the pretrained model

loss_func=torch.nn.BCELoss()
model1XLNet_Final=PreTrainedModel(model_name_XLNet,2.6785e-5,loss_func,dataset_united)


# In[20]:


# Tokenizing the entire dataset with XLNet

model1XLNet_Final.tokenize({"max_length": 200, "truncation": True, "padding": "max_length"})


# In[21]:


# Training the final XLNet model

model1XLNet_Final.train(args)


# In[ ]:


# Saving the trained model weights

model1XLNet_Final.model.save_pretrained(OUT_PATH)


# ## Training Roberta

# In[18]:


# Creating the reporting enviornment and training arguments for Roberta final model to train

wandb.init(project="FinalModel2",entity="group2tau")
OUT_PATH=r"C:\Users\user\Desktop\Group2"
roberta_args = TrainingArguments(output_dir=OUT_PATH,
                         overwrite_output_dir=True,
                         per_device_train_batch_size=17,
                         per_device_eval_batch_size=16,
                         learning_rate=1.1695e-5,
                         weight_decay=9.9237e-3,
                         save_strategy='no',
                         greater_is_better=True,
                         evaluation_strategy='epoch',
                         do_train=True,
                         num_train_epochs=3,
                         gradient_accumulation_steps=3,
                         logging_strategy='epoch',
                         warmup_steps=300,
                         report_to='wandb')


# In[19]:


# Configuring loss function and loading the pretrained model

loss_func=torch.nn.BCELoss()
model2Roberta_Final=PreTrainedModel(model_name_Roberta,2.6785e-5,loss_func,dataset_united)


# In[20]:


# Tokenizing the entire dataset with RoBerta

model2Roberta_Final.tokenize({"max_length": 200, "truncation": True, "padding": "max_length"})


# In[21]:


# Training the final RoBerta model

model2Roberta_Final.train(roberta_args)


# In[22]:


# Saving the trained model weights

model2Roberta_Final.model.save_pretrained(OUT_PATH)


# # Pruning XLNet

# In[201]:


# Creating a copy of the regular model for quantization

model1XLNet_Pruning=copy.deepcopy(model1XLNet_Final)


# In[202]:


# Showing the original parameters, with the sum of the weights at the last column

model1XLNet_Pruning.show_parameters()


# In[203]:


## Pruning 50% of the fc layers' weights

model1XLNet_Pruning.pruning(0.5)


# In[204]:


# Showing the parameters after pruning, with the sum of the weights at the last column

model1XLNet_Pruning.show_parameters()


# In[206]:


# Training the pruned model for 1 epoch, to compare accuracy

prune_args = TrainingArguments(output_dir=OUT_PATH,
                         overwrite_output_dir=True,
                         per_device_train_batch_size=10,
                         per_device_eval_batch_size=16,
                         learning_rate=2.6785e-5,
                         weight_decay=9.08e-3,
                         save_strategy='no',
                         greater_is_better=True,
                         evaluation_strategy='epoch',
                         do_train=True,
                         num_train_epochs=1,
                         gradient_accumulation_steps=6,
                         logging_strategy='epoch',
                         warmup_steps=426,
                         report_to='wandb')

model1XLNet_Pruning.train(prune_args)


# In[207]:


# Saving Pruned model Weights

model1XLNet_Pruning.model.save_pretrained(OUT_PATH)


# # XLNet Quantization

# In[209]:


# Creating a copy of the regular model for quantization

model1XLNet_Quantization=copy.deepcopy(model1XLNet_Final)


# In[210]:


# Showing model size before quantization

model1XLNet_Quantization.print_model_size()

# Applying Quantization

model1XLNet_Quantization.quantization(0)


# In[211]:


# Showing model size after quantization

model1XLNet_Quantization.print_model_size()


# In[212]:


# Saving Quantized model weights

model1XLNet_Quantization.model.save_pretrained(OUT_PATH)


# In[213]:


# Returning the quantized weights to float to compare accuracy

for i,(module,layer) in enumerate(list(model1XLNet_Quantization.model.named_modules())):
            if isinstance(layer,torch.nn.modules.linear.Linear) and 'ff' in module:
                weights=layer.weight.float()
                list(model1XLNet_Quantization.model.named_modules())[i][1].weight.data=weights


# In[214]:


# Showing the quantization

list(model1XLNet_Quantization.model.named_parameters())[-8][1]


# In[215]:


# Initiating a new trainer object to evaluate the data

evaluate_quantization=Trainer(
            model=model1XLNet_Quantization.model,
            args=args,
            train_dataset=model1XLNet_Quantization.tokenized_dataset['train'],
            eval_dataset=model1XLNet_Quantization.tokenized_dataset['test'],
            model_init=model1XLNet_Quantization.model_init,
            compute_metrics=model1XLNet_Quantization.metric_fn)


# In[218]:


# Evaluating the quantized model

model1XLNet_Quantization.model.eval()
evaluate_quantization.evaluate(model1XLNet_Quantization.tokenized_dataset['test'])

