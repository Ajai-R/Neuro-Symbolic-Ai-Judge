"""
Signal 1: InLegalBERT Fine-Tuning Pipeline (Two-Stage Layer Freezing)
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure UTF-8 output encoding on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure imports from current project root work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from signal_1_inlegalbert.dataset import RTIDataset, get_class_weights, LABEL2ID, ID2LABEL, NUM_CLASSES
from signal_1_inlegalbert.model import InLegalBERTForRTIClassification
from signal_1_inlegalbert.evaluate import evaluate_test_set


class DualLogger:
    """Simultaneously outputs to stdout and a persistent log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()

    def fileno(self):
        return self.terminal.fileno()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)



def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss'] / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_before <= scale_after:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']

            total_loss += loss.item()
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)), average='macro', zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)), average='weighted', zero_division=0
    )

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'preds': all_preds,
        'labels': all_labels
    }


def main():
    OUTPUT_DIR = os.path.join(current_dir, "checkpoints")
    BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    # Initialize dual logger to save console outputs
    log_file_path = os.path.join(OUTPUT_DIR, "train.log")
    sys.stdout = DualLogger(log_file_path)

    print("=" * 80)
    print(" SIGNAL 1: InLegalBERT NEURAL OUTCOME CLASSIFIER (TWO-STAGE LAYER FREEZING)")
    print("=" * 80)

    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")

    # 2. Optimized Hyperparameters
    MODEL_NAME = "law-ai/InLegalBERT"
    BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 1
    MAX_LENGTH = 512
    STAGE1_EPOCHS = 2   # Warmup Classifier Head (BERT Encoder 100% frozen)
    STAGE2_EPOCHS = 8   # Fine-tune Top 4 Transformer Layers (Layers 0-7 frozen)
    TOTAL_EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS

    # 3. Load Training Dataset
    train_path = os.path.join(project_root, "Dataset", "train.csv")
    print(f"\nLoading training data from: {train_path}")
    full_train_df = pd.read_csv(train_path)
    print(f"Total training cases available: {len(full_train_df)}")

    # 4. Stratified Train / Validation Split (85% Train, 15% Validation)
    train_df, val_df = train_test_split(
        full_train_df,
        test_size=0.15,
        stratify=full_train_df['outcome'],
        random_state=42
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Training split:   {len(train_df)} cases")
    print(f"Validation split: {len(val_df)} cases")

    # 5. Initialize Tokenizer & Multi-Field Prompt Datasets
    print(f"\nInitializing Tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    print("Building multi-field prompts (Query + Exemptions + Authority + Narrative)...")
    train_dataset = RTIDataset(train_df, tokenizer, max_length=MAX_LENGTH, head_len=160)
    val_dataset = RTIDataset(val_df, tokenizer, max_length=MAX_LENGTH, head_len=160)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Initialize Model with Full Balanced Class Weights (Power 1.0)
    class_weights = get_class_weights(train_df, device, smooth_power=1.0)
    print(f"Computed Class Weights: {class_weights.cpu().numpy().round(3).tolist()}")
    print(f"Loading Pretrained Model ({MODEL_NAME})...")
    model = InLegalBERTForRTIClassification(pretrained_model_name=MODEL_NAME, class_weights=class_weights)
    model.to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    csv_log_file = os.path.join(OUTPUT_DIR, "training_log.csv")
    history = []
    log_rows = []
    best_val_f1 = 0.0
    start_time = time.time()

    # =========================================================================
    # STAGE 1: WARMUP CLASSIFIER HEAD (BERT ENCODER FROZEN)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" >>> STAGE 1: WARMUP CLASSIFIER HEAD (InLegalBERT Encoder 100% Frozen)")
    print("     Goal: Align classifier head with legal embeddings without scrambling BERT")
    print("=" * 80)

    # Freeze entire BERT encoder & embeddings
    for param in model.model.bert.parameters():
        param.requires_grad = False
    for param in model.model.classifier.parameters():
        param.requires_grad = True
    if hasattr(model.model.bert, "pooler") and model.model.bert.pooler is not None:
        for param in model.model.bert.pooler.parameters():
            param.requires_grad = True

    stage1_params = [p for p in model.parameters() if p.requires_grad]
    opt_stage1 = torch.optim.AdamW(stage1_params, lr=1e-3, weight_decay=0.01)
    stage1_steps = len(train_loader) * STAGE1_EPOCHS
    sched_stage1 = get_linear_schedule_with_warmup(opt_stage1, num_warmup_steps=int(stage1_steps * 0.1), num_training_steps=stage1_steps)

    print(f"{'Epoch':<6} | {'Stage':<8} | {'Train Loss':<11} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val Macro F1':<13} | {'Val W-F1':<9} | {'Time'}")
    print("-" * 80)

    for epoch in range(1, STAGE1_EPOCHS + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, opt_stage1, sched_stage1, scaler, device, GRAD_ACCUM_STEPS)
        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['macro_f1']
        val_wf1 = val_metrics['weighted_f1']

        print(f"{epoch:<6} | {'Stage 1':<8} | {train_loss:<11.4f} | {val_loss:<10.4f} | {val_acc*100:>7.2f}% | {val_f1*100:>11.2f}% | {val_wf1*100:>7.2f}% | {epoch_time:.1f}s")

        record = {
            'epoch': epoch,
            'stage': 'Stage 1 (Head Warmup)',
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'val_accuracy': round(val_acc, 4),
            'val_macro_f1': round(val_f1, 4),
            'val_weighted_f1': round(val_wf1, 4),
            'epoch_time_sec': round(epoch_time, 2)
        }
        history.append(record)
        log_rows.append(record)
        pd.DataFrame(log_rows).to_csv(csv_log_file, index=False)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"  --> [*] New Best Model (Acc: {val_acc*100:.2f}%, Macro F1: {val_f1*100:.2f}%)! Saving checkpoint...")
            model.model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
            with open(os.path.join(BEST_MODEL_DIR, "label_mapping.json"), "w") as f:
                json.dump({"LABEL2ID": LABEL2ID, "ID2LABEL": ID2LABEL}, f, indent=2)

    # =========================================================================
    # STAGE 2: FINE-TUNE TOP 4 TRANSFORMER LAYERS (LAYERS 0-7 REMAIN FROZEN)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" >>> STAGE 2: FINE-TUNE TOP 4 LAYERS (Layers 0-7 Frozen, Layers 8-11 Unfrozen)")
    print("     Goal: Adapt high-level semantic reasoning to Indian RTI law")
    print("=" * 80)

    # Keep layers 0 to 7 frozen; unfreeze layers 8 to 11
    for layer in model.model.bert.encoder.layer[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    stage2_params = [
        {'params': model.model.bert.encoder.layer[-4:].parameters(), 'lr': 2e-5, 'weight_decay': 0.01},
        {'params': model.model.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 0.01}
    ]
    if hasattr(model.model.bert, "pooler") and model.model.bert.pooler is not None:
        stage2_params.append({'params': model.model.bert.pooler.parameters(), 'lr': 2e-5, 'weight_decay': 0.01})

    opt_stage2 = torch.optim.AdamW(stage2_params)
    stage2_steps = len(train_loader) * STAGE2_EPOCHS
    sched_stage2 = get_linear_schedule_with_warmup(opt_stage2, num_warmup_steps=int(stage2_steps * 0.1), num_training_steps=stage2_steps)

    print(f"{'Epoch':<6} | {'Stage':<8} | {'Train Loss':<11} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val Macro F1':<13} | {'Val W-F1':<9} | {'Time'}")
    print("-" * 80)

    for epoch_idx in range(1, STAGE2_EPOCHS + 1):
        global_epoch = STAGE1_EPOCHS + epoch_idx
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, opt_stage2, sched_stage2, scaler, device, GRAD_ACCUM_STEPS)
        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['macro_f1']
        val_wf1 = val_metrics['weighted_f1']

        print(f"{global_epoch:<6} | {'Stage 2':<8} | {train_loss:<11.4f} | {val_loss:<10.4f} | {val_acc*100:>7.2f}% | {val_f1*100:>11.2f}% | {val_wf1*100:>7.2f}% | {epoch_time:.1f}s")

        record = {
            'epoch': global_epoch,
            'stage': 'Stage 2 (Top-4 Layers)',
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'val_accuracy': round(val_acc, 4),
            'val_macro_f1': round(val_f1, 4),
            'val_weighted_f1': round(val_wf1, 4),
            'epoch_time_sec': round(epoch_time, 2)
        }
        history.append(record)
        log_rows.append(record)
        pd.DataFrame(log_rows).to_csv(csv_log_file, index=False)

        # Checkpoint based on Validation Macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"  --> [*] New Best Model (Acc: {val_acc*100:.2f}%, Macro F1: {val_f1*100:.2f}%)! Saving checkpoint...")
            model.model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
            with open(os.path.join(BEST_MODEL_DIR, "label_mapping.json"), "w") as f:
                json.dump({"LABEL2ID": LABEL2ID, "ID2LABEL": ID2LABEL}, f, indent=2)

    total_training_time = time.time() - start_time
    print("=" * 80)
    print(f"\n[DONE] Two-Stage Training Completed in {total_training_time:.1f} seconds ({total_training_time/60:.1f} mins).")
    print(f"Peak Validation Macro F1: {best_val_f1*100:.2f}%")

    # Save complete JSON history
    history_file = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_file, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "stage1_epochs": STAGE1_EPOCHS,
            "stage2_epochs": STAGE2_EPOCHS,
            "batch_size": BATCH_SIZE,
            "best_val_macro_f1": best_val_f1,
            "total_time_sec": total_training_time,
            "history": history
        }, f, indent=2)
    print(f"Training history saved to: {history_file}")
    print(f"Epoch-by-epoch CSV log saved to: {csv_log_file}")
    print(f"Best model checkpoint saved to: {BEST_MODEL_DIR}")

    # Automatically run test benchmark evaluation
    print("\n" + "=" * 80)
    print(" RUNNING AUTOMATIC TEST BENCHMARK EVALUATION (UNSEEN BLIND TEST SET)")
    print("=" * 80)
    evaluate_test_set()


if __name__ == "__main__":
    main()
