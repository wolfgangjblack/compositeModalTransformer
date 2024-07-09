import copy
import torch
import numpy as np
from tqdm import tqdm
from safetensors.torch import save_file
from sklearn.metrics import classification_report

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, target_class=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    macro_accuracy = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    
    class_metrics = {f"class_{i}": {"accuracy": report[str(i)]['precision'], "f1": report[str(i)]['f1-score']} 
                     for i in range(len(report) - 3)}  # -3 to exclude 'accuracy', 'macro avg', and 'weighted avg'
    
    target_metric = None
    if target_class is not None:
        target_metric = class_metrics[f"class_{target_class}"]['f1']
    
    return avg_loss, macro_accuracy, macro_f1, class_metrics, target_metric

### NEED TO REVIEW BELOW
import torch
from safetensors.torch import save_file, load_file
import copy
from contextlib import contextmanager

@contextmanager
def evaluation_mode(model):
    model.eval()
    yield
    model.train()

def train_model(model, num_epochs, train_dataloader, eval_dataloader, optimizer, criterion, device, target_class, early_stopping=True, verbose=True):
    best_target_metric = float('-inf')
    best_model_weights = None
    history = {'train_loss': [], 'eval_loss': [], 'accuracy': [], 'f1': [], 'target_metric': []}
    
    if early_stopping:
        early_stopping = EarlyStopping(patience=5, verbose=verbose, path='best_model.safetensors')

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        
        with evaluation_mode(model):
            eval_loss, accuracy, f1, class_metrics, target_metric = evaluate(model, eval_dataloader, criterion, device, target_class)
        
        history['train_loss'].append(train_loss)
        history['eval_loss'].append(eval_loss)
        history['accuracy'].append(accuracy)
        history['f1'].append(f1)
        history['target_metric'].append(target_metric)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
            print(f"Macro Accuracy: {accuracy:.4f}, Macro F1 Score: {f1:.4f}")
            for class_name, metrics in class_metrics.items():
                print(f"{class_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            print(f"Target Class ({target_class}) F1: {target_metric:.4f}")
        
        if target_metric > best_target_metric:
            best_target_metric = target_metric
            best_model_weights = copy.deepcopy(model.state_dict())
            save_file(best_model_weights, 'best_model.safetensors')
        elif verbose:
            print(f"Target metric did not improve. Reverting to best weights.")
        
        model.load_state_dict(load_file('best_model.safetensors'))
        
        if early_stopping:
            early_stopping(eval_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    model.load_state_dict(load_file('best_model.safetensors'))

    with evaluation_mode(model):
        final_loss, final_accuracy, final_f1, final_class_metrics, _ = evaluate(model, eval_dataloader, criterion, device)
    
    if verbose:
        print("Final Evaluation Results:")
        print(f"Loss: {final_loss:.4f}, Macro Accuracy: {final_accuracy:.4f}, Macro F1 Score: {final_f1:.4f}")
        for class_name, metrics in final_class_metrics.items():
            print(f"{class_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    save_file(model.state_dict(), 'final_model.safetensors')
    return model, history