import os
import torch

from src.data import CustomImageTextDataset
from transformers import TrainingArguments, Trainer
from transformers.tokenization_utils_base import BatchEncoding
from src.model import MultimodalConfig, MultimodalProcessor, MultimodalModel

## Get composite pretrained models

###models are also at https://huggingface.co/wolfgangblack/multimodalComposite/tree/main
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
    if 'labels' in out:
        out['labels'] = torch.tensor(out['labels'])
    elif 'label' in out:
        out['labels'] = torch.tensor(out['label'])
        del out['label']  # Remove the original 'label' key    
    return out

## instantiate model
model = MultimodalModel(config)
def compute_metrics(eval_pred):
    print("Computing metrics...")
    try:
        logits, labels = eval_pred
        if isinstance(labels, dict):
            labels = labels['label']  # Adjust this key if necessary
        print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
        predictions = np.argmax(logits, axis=-1)
        eval_loss = np.mean(logits - labels)
        accuracy = np.mean(predictions == labels)
        f1 = f1_score(labels, predictions, average='weighted')
        print(f"Computed metrics - Accuracy: {accuracy}, F1: {f1}")
        return {
            "eval_acc": accuracy,
            "eval_f1": f1,
            "eval_loss": eval_loss
        }
    except Exception as e:
        print(f"Error in compute_metrics: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
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
    save_safetensors=True,  # Ensure this is supported by your Transformers version
    metric_for_best_model="eval_f1",  # Choose eval_f1 or eval_acc
    greater_is_better=True,  # Set to True because higher F1/accuracy is better
    report_to='all',  # Enable detailed logging to all available loggers
    disable_tqdm=False,  # Ensure progress bars are shown
    logging_first_step=True,  # Log the first step

)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=validation_data,
    data_collator=custom_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
