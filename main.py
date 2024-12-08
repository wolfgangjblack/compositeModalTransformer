import os
import torch
import logging
import yaml
from datetime import datetime
from src.data import CustomImageTextDataset
from src.training import train_model, evaluate
from transformers.tokenization_utils_base import BatchEncoding
from src.model import MultimodalConfig, MultimodalProcessor, MultimodalModel

# Load configuration from a YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set up logging
log_dir = config['log_dir']
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_filename = os.path.join(log_dir, f"{timestamp}_{config['log_filename']}.log")
logging.basicConfig(filename=log_filename, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

logging.info("Starting the training script")

# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    logging.info("Using macOS Metal")
else:
    device = torch.device("cpu")
    logging.info("Using CPU")

# Get composite pretrained models
model_paths = {i: os.path.join('models', i) for i in os.listdir('models') if 'yoloRater' not in i}

# Create Config
model_config = MultimodalConfig(models=model_paths)

# Create dataset
training_data = CustomImageTextDataset(config['train_data_dir'], target_transform=model_config.label2id)
validation_data = CustomImageTextDataset(config['val_data_dir'], target_transform=model_config.label2id)

# Instantiate processor
processor = MultimodalProcessor(models=model_paths)

# Custom collator
def custom_collator(batch):
    img_paths = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    tags = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    
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

# Instantiate model
model = MultimodalModel(model_config)

# Set up optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
criterion = torch.nn.CrossEntropyLoss()

# Train the model
logging.info("Starting training")
train_model(
    model,
    config['num_epochs'],
    training_data,
    validation_data,
    optimizer,
    criterion,
    device,
    config['target_class'],
    config['batch_size']
)

# Final evaluation
logging.info("Evaluating the model")
final_loss, final_accuracy, final_f1, final_class_metrics, _ = evaluate(model, validation_data, criterion, device)
logging.info("Final Evaluation Results:")
logging.info(f"Loss: {final_loss:.4f}, Macro Accuracy: {final_accuracy:.4f}, Macro F1 Score: {final_f1:.4f}")
for class_name, metrics in final_class_metrics.items():
    logging.info(f"{class_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
