import os
import torch

from src.data import CustomImageTextDataset
from transformers import TrainingArguments, Trainer
from transformers.tokenization_utils_base import BatchEncoding
from src.model import MultimodalConfig, MultimodalProcessor, MultimodalModel

## Get composite pretrained models
model_paths = {i: os.path.join('models', i) for i in os.listdir('models') if 'yoloRater' not in i}

## Create Config
config = MultimodalConfig(models = model_paths)

##Create dataset
datadir = "./data/datasets/mayWithTags/train"
valdir = "./data/datasets/mayWithTags/val"

training_data = CustomImageTextDataset(datadir, target_transform=config.label2id)
validation_data = CustomImageTextDataset(valdir, target_transform=config.label2id)

##Instantiate processor, this takes raw data and formats it for the various models
processor = MultimodalProcessor(models = model_paths)

## We need the data collator for the trainer api
## Why is this not in src.data? we need to use the processor to process the data
## We COULD always use a lambda function to pass the processor to the collator
# but that seems like extra work
def custom_collator(batch):
    img_paths = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    tags = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    
    # Assuming processor is defined elsewhere and processes the data correctly
    processed_batch = [processor(img_paths[i], texts[i], tags[i], labels[i]) for i in range(len(img_paths))]
    
    out = {}
    
    for key in processed_batch[0].keys():
        if isinstance(processed_batch[0][key], torch.Tensor):
            out[key] = torch.stack([i[key] for i in processed_batch])
        elif isinstance(processed_batch[0][key], BatchEncoding):

            tmp = [i[key] for i in processed_batch]
           
            var = {k: [] for k in tmp[0].keys()}
            for k in var.keys():
                var[k] = torch.stack([i[k] for i in tmp])
                
            out[key] = var
        else:
            out[key] = [i[key] for i in processed_batch]
    
    return out

## instantiate model
model = MultimodalModel(config)

## instantiate trainer
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
    fp16=True,                       # Enable mixed precision training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=validation_data,
    data_collator=custom_collator
)
