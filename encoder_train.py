import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DefaultDataCollator
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import argparse
from collections import Counter
from transformers import DataCollatorWithPadding

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='')
    
# Adding arguments
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--experiment', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

model_name = args.model_name
experiment = args.experiment
batch_size = args.batch_size


if experiment == "evasion_based_clarity":
    num_labels = 9
    mapping_labels = {
        'Explicit': 0,
        'Implicit': 1,
        'Dodging': 2,
        'Deflection': 3,
        'Partial/half-answer': 4,
        'General': 5,
        'Declining to answer': 6,
        'Claims ignorance': 7,
        'Clarification': 8
    }
    label = "evasion_label"
elif experiment == "direct_clarity":
    num_labels = 3
    mapping_labels = {'Clear Reply': 0, "Ambivalent": 1, "Clear Non-Reply": 2}
    label = "clarity_label"


# --- Load Model and Tokenizer ---
print(f"Loading model and tokenizer for: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

max_size = 512

print(f"Model {model_name} loaded. Max sequence length set to {max_size}.")

# Example data

dataset = load_dataset("ailsntua/QEvasion")
dataset = dataset.filter(lambda x: x[label] != '')

def tokenize_function(examples):
    inputs = [q + " " + a for q, a in zip(examples["interview_question"], examples["interview_answer"])]
    
    tokenized_inputs = tokenizer(
        inputs, 
        truncation=True,  
        max_length=max_size
    )

    tokenized_inputs["labels"] = [mapping_labels[str_label] for str_label in examples[label]]

    return tokenized_inputs

train_data_raw = dataset['train']

tokenized_dataset = train_data_raw.map(
    tokenize_function, 
    batched=True, 
    num_proc=4, # Use 4 processes for tokenization
    remove_columns=[col for col in dataset["train"].column_names if col not in [label, "ID"]]
)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("Splitting dataset into 90% Train and 10% Validation...")
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = split["train"]
val_dataset = split["test"]

print(f"Train Size: {len(train_dataset)}")
print(f"Val Size: {len(val_dataset)}")

data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,
    collate_fn=data_collator
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    collate_fn=data_collator
)

print(f"DataLoaders ready. Training batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

all_labels = [mapping_labels[row[label]] for row in dataset['train']]
label_counts = Counter(all_labels)
print(label_counts)
num_labels = len(mapping_labels)
total = sum(label_counts.values())

class_weights_list = []
for i in range(num_labels):
    count = label_counts[i]
    if count > 0:
        weight = total / count
    else:
        weight = 1.0 # Default weight for missing classes (prevents crash)
    class_weights_list.append(weight)

class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float)

model.to(device)
class_weights = class_weights_tensor.to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

print("Class weights:", class_weights)

print(len(train_dataloader), len(val_dataloader))

# Fine-tuning
optimizer = AdamW(model.parameters(), lr=1e-5)

num_epochs = 10

out_file = f"{model_name.split('/')[-1]}-qaevasion-{experiment}"
best_macro_f1 = 0.0

patience = 3
epochs_without_improvement = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
    
        outputs = model(input_ids=inputs, attention_mask=attention_mask)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Inside the validation loop
    model.eval()
    val_loss = 0.0
    
    pred_labels = [] 
    true_labels = [] 
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
    
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
    
            # Calculate accuracy
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            pred_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    average_val_loss = val_loss / len(val_dataloader)
    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')

    print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {average_val_loss:.4f} - Accuracy: {accuracy * 100:.2f}% - Macro F1 Score: {macro_f1:.4f}')