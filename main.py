import os
import torch
from src.data import CustomImageTextDataset
from src.training import train_model, evaluate
from transformers.tokenization_utils_base import BatchEncoding
from src.model import MultimodalConfig, MultimodalProcessor, MultimodalModel

## Get composite pretrained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    return out

## instantiate model
model = MultimodalModel(config)
    

#### Need to check below

# Set up optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# Train the model
num_epochs = 10
target_class = 2  # Prioritize F1 for class 2
train_model(model, num_epochs, train_dataloader, eval_dataloader, optimizer, criterion, device, target_class)

# Final evaluation
final_loss, final_accuracy, final_f1, final_class_metrics, _ = evaluate(model, eval_dataloader, criterion, device)
print("Final Evaluation Results:")
print(f"Loss: {final_loss:.4f}, Macro Accuracy: {final_accuracy:.4f}, Macro F1 Score: {final_f1:.4f}")
for class_name, metrics in final_class_metrics.items():
    print(f"{class_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
