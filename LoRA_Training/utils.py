import accelerate
import datasets
import evaluate
import math
import numpy as np
import os
import peft
import pickle
import pytest, ipytest
#ipytest.autoconfig()
import pandas as pd
import transformers
import torch
import time
import threading
from datasets import(
    load_dataset, 
    load_dataset_builder,
    get_dataset_split_names,
    get_dataset_config_names,
    Value
)


from peft import(
    LoftQConfig,
    LoraConfig,
    get_peft_model,
    PeftModelForCausalLM
)

from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TrainingArguments,
    Trainer
)
from trl import SFTTrainer, SFTConfig
# allows fast processing of datasets

def model_from_pkl(model):
    file = "pkl_files/" + model + ".pkl"
    #if os.path.exists(file)==False:
        #%run make_models.ipynb # create the pkl files if they don't exist
    with open("pkl_files/" + model + ".pkl", "rb") as f:
        pkl_model=pickle.load(f)
    model_name=pkl_model["model_name"]
    #model_name.device_map="auto"
    tokenizer=pkl_model["tokenizer"]
    tokenizer.pad_token=tokenizer.eos_token
    return model_name, tokenizer


ds_gst1_train=load_dataset("LongSafari/open-genome", "stage1", split="train[:500]")
#print(ds_gst1[50])
ds_gst1_test=load_dataset("LongSafari/open-genome", "stage1", split="test[:50]")
print(get_dataset_split_names("LongSafari/open-genome", "stage1"))
ds_gst2_train=load_dataset("LongSafari/open-genome", "stage2", split="train[:500]")
ds_gst2_test=load_dataset("LongSafari/open-genome", "stage2", split="test[:50]")
print(get_dataset_split_names("LongSafari/open-genome", "stage2"))

def search_with_strings(training_data, base_pair):
    present_tally=0 # running tally of how many times a string occurs in the training data
    final_tally=0 # tally to be returned of how many times the longest string occurs
    temp_string="" # tracks what characters are present in the training data
    present_string="" # final string, max consecutive present characters
    data_len=0
    for character in base_pair: # for each character in the imported base pair
        temp_string+=character # add the character to the temp string
        for data in training_data: # for each item in the training data
            if temp_string in data['text']: # if the temp string is present
                present_tally+=1 # add a tally for each time it is present
                final_tally=present_tally # save the running tally
        if present_tally==0: # present_tally is 0, this string is no longer present
            data_len=len(temp_string)-1 # the length of the present string
            present_string=temp_string[:-1] # return the present string (remove the character that wasn't present)
            return present_string, data_len, final_tally
            #return the longest string in the training data, the length of the string, and how many times the string appears
        present_tally=0 # reset the present tally each time you add a character to the string

    # this code block runs if the entire imported string of base pairs is present in the training data
    present_string=temp_string  # the entire temp_string is present
    data_len=len(present_string) 
    return present_string, data_len, final_tally
    #return the whole string, its length, and how many times the string appears
            
# perform preprocessing on the genomic data
def map_data(data, model, tokenizer):
    def tokenize_l_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)
    def tokenize_m_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)
        
    if type(model)==transformers.models.llama.modeling_llama.LlamaForCausalLM:
        tokenized_dataset=data.map(tokenize_l_function, batched=True)
    else:
        tokenized_dataset=data.map(tokenize_m_function, batched=True)
    return tokenized_dataset


# perform preprocessing on the genomic data
def dp_tokenizer(tokenizer):
     if type(model)==transformers.models.llama.modeling_llama.LlamaForCausalLM:
        return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors='pt').input_ids.to(device)
     else:
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024,return_tensors='pt').input_ids.to(device)
        # return_tensors=pt ensures a pytorch tensor is returned, which is needed for a custom training loop
        # input_ids.to(device) ensures this will run on the GPU

# before loading in the base model with LoRA, might be good to define a helper function
# this looks at the total parameters a model has, and how many are trainable
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

# A trainer needs to be passed a function from the Evaluate library (specifically the accuracy function) to compute and report metrics
metric=evaluate.load("accuracy")

# these functions are used to track if a function is still running
# threads allow us to do multiple tasks at once, this runs in the background during training
def initialize_heartbeat():
    running=threading.Event() # create an event
    running.set() # set the running event to true, meaning the process will execute
    process = threading.Thread(target=heartbeat, args=(running,)) # initialize the thread
    process.start() # start the process
    return running # return the event for later use

def heartbeat(running):
    while running.is_set(): # while the running event is true
        print("The function is running")
        time.sleep(45) # "The function is running" will print every 45 seconds

def end_heartbeat(running):
    running.clear() #running.clear() makes the event false, ending the process


# the compute_metrics method will calculate prediction accuracy
comp_metrics_output=[]
def compute_metrics(eval_pred):
    state=initialize_heartbeat() # start the thread, so when this function begins we have a heartbeat
    start_time=time.monotonic() # get the time at the start of the function
    logits=eval_pred.predictions
    refs=eval_pred.label_ids
    log_32=logits.astype(np.int32)
    log_32=np.concatenate(log_32).tolist()
    ref_32=refs.astype(np.int32)
    ref_32=np.concatenate(ref_32).tolist()
    predictions = np.argmax(log_32, axis=-1)
    met=metric.compute(predictions=predictions, references=ref_32)
    comp_metrics_output.append(met)
    end_time=time.monotonic() # get time at the end
    duration=end_time-start_time # subtract end time from smart time to get how long evaluation took
    print("The time taken was", duration, " seconds")
    end_heartbeat(state) # end the heartbeat thread
    return met
    # this lets us convert logits (returned by models) into predictions
    # np.argmax returns the indices of the maximum values along the axis of an array
    # axis=-1 means it looks at the last axis in the array
    # metric.compute gathers all cached predictions and references to compute the metric score

# the trainer object specifies the model, training arguments, training and test datasets, and evaluation function
def make_trainer(m_model, train_data, test_data, config, args, optimizer=None):
    trainer=SFTTrainer(
        model=m_model,
        train_dataset=train_data,
        eval_dataset=test_data,
        peft_config=config,
        args=args,
        #compute_metrics=compute_metrics,
        compute_metrics=None,
        optimizers=([optimizer, None]),
        )
    return trainer
    
# SFTTrainer is best used for training with a pre-trained model and a smaller dataset
# It can be better suited for fine-tuning than regular Trainer

def get_dataframe(training_output: list, strategy):
    df=pd.DataFrame(training_output) # convert the imported list of dictionaries to a DataFrame
    df.index=df[strategy] # the index of the dataframe is whatever evaluation strategy was used
    df=df.drop([strategy], axis=1) # drop one column so there aren't two step/epoch columns
    df.plot(y=0, xlabel=strategy, ylabel="Training Loss", title="Fine-Tuning Training Evaluation") # plot training loss
    df.plot(y=1, xlabel=strategy, ylabel="Validation Loss", title="Fine-Tuning Validation Evaluation") # plot validation loss
    if 'eval_accuracy' in training_output:
        df.plot(y=2, xlabel=strategy, ylabel="Accuracy", title="Fine-Tuning Accuracy Evaluation") # plot accuracy
    #for all of the above plots, the evaluation strategy (the index) is the x-axis value
    return df

def get_training_output(trainer, keys: list):
    trainer_info=[]
    temp_dict={}
    logs=trainer.state.log_history # get the logs from model training, these show training loss, accuracy, etc
    strat=trainer.args.eval_strategy.value # was this evaluated at steps or epochs
    def check_eval(strat, log):
        condition=False
        state=0
        if (strat=='epoch'):
            condition= log['epoch'].is_integer() 
            state=log['epoch'] # save the epoch number
            # if evaluated at epochs, extract data at the points where epochs are whole numbers
        elif (strat=='steps'):
            val=math.floor(log['step']) # use floor to convert the floating point step to an integer
            condition = val%trainer.args.logging_steps==0
            state=val # save the step number
            # if evaluated at steps, evaluate at the point where the number of steps divides evenly by the training interval
        return condition, state
        
    for log in logs: # loop through training logs
        condition, state=check_eval(strat, log)
        if not (condition): # check the appropriate condition based on evaluation strategy
            continue # whenever the condition isn't true, restart the loop
        for key in keys: # look at all the keys (usually training loss, validation loss, and accuracy)
            if key in log:
                temp_dict[key]=log[key] # at the value tied to each key to a placeholder dictionary
        if key in temp_dict: # if a key is already in the dictionary (you've found a value for a different step/epoch)
            temp_dict[strat]=state # add the corresponding step/epoch number to the temp dictionary
            trainer_info.append(temp_dict) # add the temp dictionary to the list with training information
            temp_dict={} # clear the temp dictionary, new values with the same keys as the last can now be added
    training_output=get_dataframe(trainer_info, strat) # run this method to convert the list of dicts to a dataframe
    return training_output #return the dataframe

keys=list(globals().keys())
keys
for key in keys:
    if key[0]!='_':
        print(f"{key},", end=" ")
del keys
del key