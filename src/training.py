import copy
import torch
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
from safetensors.torch import save_file, load_file
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

@contextmanager
def evaluation_mode(model):
    model.eval()
    yield
    model.train()

def train_model(model,num_epochs, train_dataloader, eval_dataloader,
                optimizer, criterion, device, target_class,
                early_stopping=True, verbose=True):
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


class CustomTrainer(Trainer):
    def __init__(self, device=torch.device("mps" if torch.torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu"), *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure model is on the correct device
        self.model.to(device)

        # Create optimizer and scheduler
        self.create_optimizer_and_scheduler(num_training_steps=self.args.max_steps)

    def _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval):
        # Ensure model is on the correct device
        self.model.to(self.model.device)

        # Initialize tr_loss on the correct device
        tr_loss = torch.tensor(0.0).to(self.model.device)
        self.model.zero_grad()
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        model = self.model

        self.state.epoch = 0
        self.state.global_step = 0

        for epoch in range(int(args.num_train_epochs)):
            correct_predictions = 0
            total_predictions = 0
            epoch_steps = len(self.get_train_dataloader())
            for step, inputs in enumerate(self.get_train_dataloader()):
                tr_loss_step = self.training_step(model, inputs)
                tr_loss_step = tr_loss_step.to(self.model.device)  # Ensure tr_loss_step is on the correct device
                tr_loss += tr_loss_step
                self.state.global_step += 1

                # Calculate accuracy
                outputs = self.model(**inputs)
                logits = outputs["logits"]
                labels = inputs["labels"]
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                if (step + 1) % args.logging_steps == 0:
                    accuracy = correct_predictions / total_predictions
                    logs = {
                        "loss": tr_loss.item() / (step + 1),
                        "accuracy": accuracy,
                        "step": step + 1,
                        "epoch_steps": epoch_steps
                    }
                    self.log(logs)

                if (step + 1) % args.eval_steps == 0:
                    self.evaluate()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    step + 1 == len(self.get_train_dataloader())
                ):
                    if args.fp16 and _use_native_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

            self.state.epoch += 1
            self.evaluate()

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        return self._finalize_train(model, tr_loss)

    def evaluate(self):
        eval_dataloader = self.get_eval_dataloader()
        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=False
        )

        # Log evaluation metrics
        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control)

        return output.metrics

    def log(self, logs):
        logs["epoch"] = self.state.epoch
        logs["step"] = self.state.global_step
        self.state.log_history.append(logs)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control)
        if self.is_world_process_zero():
            print(logs)