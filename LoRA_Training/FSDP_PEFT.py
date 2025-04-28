from utils import * # get all the functions used in this file
from accelerate import Accelerator, DistributedType
#import argparse

import torch
import torch.nn as nn
from opacus.grad_sample import GradSampleModule
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
#import torch.multiprocessing as mp

# from torch.distributed.fsdp import (
#    FullyShardedDataParallel,
#    CPUOffload,
# )
# from torch.distributed.fsdp.wrap import (
#     size_based_auto_wrap_policy,
#     enable_wrap,
#     wrap,
# )



lla_321, lla_321_tokenizer=model_from_pkl("Llama-3.2-1B-Instruct")

ds_gst1_train=load_dataset("LongSafari/open-genome", "stage1", split="train[:500]")
#print(ds_gst1[50])
ds_gst1_test=load_dataset("LongSafari/open-genome", "stage1", split="test[:50]")
print(get_dataset_split_names("LongSafari/open-genome", "stage1"))
ds_gst2_train=load_dataset("LongSafari/open-genome", "stage2", split="train[:500]")
ds_gst2_test=load_dataset("LongSafari/open-genome", "stage2", split="test[:50]")
print(get_dataset_split_names("LongSafari/open-genome", "stage2"))
# using a smaller dataset to make sure this works

def preprocess_data(dataset):
    dataset=dataset.remove_columns(["text", "record"]) # pytorch does not accept this input
    
    dataset.set_format("torch", columns=["input_ids", "attention_mask"], dtype=torch.long) # ensures Tensors are returned
    return dataset

#with Accelerator.main_process_first():
l_tokenized_stage1_train=map_data(ds_gst1_train, lla_321, lla_321_tokenizer)
l_tokenized_stage1_test=map_data(ds_gst1_test, lla_321, lla_321_tokenizer)

l_processed_train=preprocess_data(l_tokenized_stage1_train)
l_processed_test=preprocess_data(l_tokenized_stage1_test)

# for batch in l_processed_train:
#     print((batch['attention_mask'].dtype))
from torch.utils.data import DataLoader
train_dataloader=DataLoader(l_processed_train, shuffle=False, batch_size=1, pin_memory=True)
test_dataloader=DataLoader(l_processed_test, shuffle=False, batch_size=1, pin_memory=True)

config=LoraConfig(
    #inference_mode=False,
    r=8, #rank of update matrices, lower value results in smaller matrices with fewer parameters
    lora_alpha=16, #LoRA scaling factor
    #target_modules=["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.1, # dropout probability of LoRA layers
    bias="none", # specifies if bias parameters should be trained
    #modules_to_save=["decode_head"] #models apart from LoRA layers that are trainable
)

lla_lora_model=get_peft_model(lla_321, config)
# print_trainable_parameters(lla_lora_model)
#lla_lora_model=lla_321

tally=0
for name, parameter in lla_lora_model.named_parameters():
    tally+=1

print(tally)


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")



from transformers import get_scheduler
from accelerate.test_utils.testing import get_backend
def dp_train():
    accelerator = Accelerator()
    optimizer=torch.optim.AdamW(lla_lora_model.parameters(),
                           amsgrad=False, # the AMSGrad variant of this algorithm won't be used 
                            betas=(0.9, 0.999), # coefficients used for computing running averages of gradient and its square
                            capturable=False, # whether the instance will be captured in a CUDA graph
                            differentiable=False, # whether autogad should occur through the optimzer step in training
                            eps=1e-08, # added to denominator to improve numerical stablitity
                            foreach=None, # whether foreach implementation is used
                            fused=None, #whether the fused implementation is used
                            #initial_lr=2e-05,
                            lr=0.1, #learning rate
                            maximize=False, # whether the object is maximized with respect to params instead og
                            weight_decay=0.0)

    l3_trainer=make_trainer(lla_lora_model, train_dataloader.dataset, test_dataloader.dataset, config,
                          SFTConfig(output_dir="test_trainer", eval_strategy="epoch",
                                    per_device_train_batch_size=1,
                                    max_grad_norm=1.0,
                                    num_train_epochs=1,
                                    logging_strategy="epoch",
                                    #logging_steps=6
                                    fp16=True))
    #print(optimizer)
    num_epochs=l3_trainer.args.num_train_epochs
    num_steps=num_epochs * len(train_dataloader)
    lr_scheduler=get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=num_steps,
            ) #this is the default learning rate scheduler from the trainer
    print(type(l3_trainer.model))
    if type(l3_trainer.model):
         l3_trainer.model.print_trainable_parameters()
    if getattr(l3_trainer.accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin = l3_trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(l3_trainer.model)
    model=l3_trainer.model
    
    model, optimizer, tr_dataloader, te_dataloader, scheduler = accelerator.prepare(
         model, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )
    model.to(accelerator.device)
    ddp_model=DDP(model, device_ids=[accelerator.device])
    scaler = torch.cuda.amp.GradScaler() # used for mixed precision training
    
   
    print(f"{num_steps=}")
    for epoch in range(int(num_epochs)):
        model.train()
        for step, batch in enumerate(tr_dataloader):
            optimizer.zero_grad()
            #with torch.autocast(device_type='cuda', dtype=torch.float16):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            loss=ddp_model(**batch).loss
    #         #print(model)
            #loss = outputs.loss

            # scaling loss, 
            #backward is called on scaled loss to get scaled gradients
            #scaler.scale(loss).backward()
            accelerator.backward(loss)
            # gradients are unscaled before the optimizer is called
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # # add if statement for steps/multiple epochs here
            # eval_loss, eval_accuracy = trainer.evaluate()
            # print(eval_loss)
            # Clear cache if necessary
            torch.cuda.empty_cache()
    print("Training complete")
    #end_heartbeat(state)
    

# def run_training(fn, world_size):
#     mp.spawn(fn,
#              args=(world_size,),
#              nprocs=world_size,
#              join=True)
def main():
    ddp_setup()
    # world_size=torch.cuda.device_count()
    #state=initialize_heartbeat()
    dp_train()
    dist.destroy_process_group()
    #end_heartbeat(state)


if __name__=="__main__":
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'
    main()

