import os

from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import KFold
import random
import numpy as np
import nltk
import nlpaug.augmenter.word as naw
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup
)
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('popular')
nltk.download('averaged_perceptron_tagger_eng')


def truncate(example):
    return {
        'text': " ".join(example['text'].split()[:100]),
        'label': example['label']
    }


def data_preprocess(dataset_path):
    health_ds = load_dataset('json', data_files=dataset_path)["train"]

    remove_labels = [2, 3, 5, 7]
    filtered_ds = health_ds.filter(lambda example: example['label'] not in remove_labels)
    unique_labels = sorted(set(filtered_ds['label']))
    label_map = {old: new for new, old in enumerate(unique_labels)}

    def remap_label(example):
        return {'label': label_map[example['label']]}

    health_ds = filtered_ds.map(remap_label)

    aug_health_ds = data_augment(health_ds)
    dataset_dict = aug_health_ds.train_test_split(test_size=0.2, seed=1111)

    return DatasetDict({
        "train": dataset_dict["train"].shuffle(seed=1111),
        "val": dataset_dict["test"]
    })


def data_augment(dataset):
    augmenter = naw.SynonymAug(aug_min=1, aug_max=3)

    original_texts = dataset['text']
    original_labels = dataset['label']
    aug_texts = []
    aug_labels = []

    for text, label in zip(original_texts, original_labels):
        augmented_versions = augmenter.augment(text, n=2)
        if isinstance(augmented_versions, str):
            augmented_versions = [augmented_versions]
        aug_texts.extend(augmented_versions)
        aug_labels.extend([label] * len(augmented_versions))

    all_texts = list(original_texts) + aug_texts
    all_labels = list(original_labels) + aug_labels

    return Dataset.from_dict({
        'text': all_texts,
        'label': all_labels
    })


def model_preprocess(model_name, dataset):
    if model_name == "bert":
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = (DistilBertForSequenceClassification
                 .from_pretrained("distilbert-base-uncased",
                                  num_labels=5))
        for param in model.distilbert.parameters():
            param.requires_grad = False
        model.to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    tokenized_dataset = dataset.map(
        lambda token: tokenizer(token["text"], padding="max_length", truncation=True),
        batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return model, tokenized_dataset


def train_val(model, tokenized_dataset, epochs=3, bsz=8, lr=5e-3):
    train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=bsz, shuffle=True)
    test_dataloader = DataLoader(tokenized_dataset["val"], batch_size=bsz)

    optimizer = AdamW(model.parameters(), lr=lr)
    num_warmup_steps = int(0.1 * epochs * len(train_dataloader))

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=epochs * len(train_dataloader))

    train_acc_record = []
    train_loss_record = []
    test_acc_record = []

    for epoch in tqdm(range(epochs), desc="Train and Test", leave=True):

        model.train()
        train_correct, train_total = 0, 0
        for ids, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            if 'label' in batch:
                batch['labels'] = batch.pop('label')
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            if ids % 8 == 0:
                train_loss_record.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            predictions = torch.argmax(logits, dim=-1)
            train_correct += (predictions == batch["labels"]).sum().item()
            train_total += batch["labels"].size(0)

        train_acc = train_correct / train_total

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                if 'label' in batch:
                    batch['labels'] = batch.pop('label')
                outputs = model(**batch)
                logits = outputs.logits

                predictions = torch.argmax(logits, dim=-1)
                test_correct += (predictions == batch["labels"]).sum().item()
                test_total += batch["labels"].size(0)

        test_acc = test_correct / test_total

        train_acc_record.append(train_acc)
        test_acc_record.append(test_acc)

        tqdm.write(f"Epoch {epoch + 1}: train acc = {train_acc:.4f}, test acc = {test_acc:.4f}")
    return train_acc_record, test_acc_record, train_loss_record


if __name__ == '__main__':
    dataset_path = "data/healthcare-consults.json"
    model_name = "bert"
    num_epochs = 9
    bsz = 3
    lr = 5e-3
    os.makedirs("output", exist_ok=True)

    health_ds = data_preprocess(dataset_path)
    model, tokenized_dataset = model_preprocess(model_name, health_ds)
    train_acc, test_acc, train_loss = train_val(model, tokenized_dataset, num_epochs, bsz, lr)
    torch.save(model, f"output/{model_name}_model.pth")
    print(f"{model_name} parameter is successfully saved")
