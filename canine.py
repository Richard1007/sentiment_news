import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer,
    TrainingArguments
)


def encode_data(dataset, tokenizer, max_seq_length=128):
    tokenized_examples = tokenizer(
        dataset["Title"].tolist(),
        truncation=True,
        max_length=max_seq_length,
        padding="max_length"
    )

    input_ids = torch.Tensor(tokenized_examples['input_ids']).long()
    attention_mask=torch.Tensor(tokenized_examples['attention_mask']).long()
    return input_ids,attention_mask


def extract_labels(dataset):
# Converts labels into numerical labels and returns as a list
    labels=[]
    label_df=dataset["label"].tolist()
    for i in range(len(label_df)):
        if label_df[i]==1:
            labels.append(1)
        else:
            labels.append(0)
    return labels


class DebertaDataset(Dataset):
    """
    A torch.utils.data.Dataset wrapper for the dataset.
    """

    def __init__(self, dataframe, tokenizer, max_seq_length=256):
        """
        Args:
          dataframe: A Pandas dataframe containing the data.
          tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
            tokenize the data.
          max_seq_length: Maximum sequence length to either pad or truncate every
            input example to.
        """
        ## TODO: Use encode_data() from data_utils to store the input IDs and 
        ## attention masks for the data.
        self.encoded_data = encode_data(dataframe, tokenizer, max_seq_length)
        self.label_list = extract_labels(dataframe)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, i):
        """
        Returns:
          example: A dictionary containing the input_ids, attention_mask, and
            label for the i-th example, with the values being numeric tensors
            and the keys being 'input_ids', 'attention_mask', and 'labels'.
        """
        ## TODO: Return the i-th example as a dictionary with the keys and values
        ## specified in the function docstring. You should be able to extract the
        ## necessary values from self.encoded_data and self.label_list.
        item_dict={}
        item_dict["input_ids"]=self.encoded_data[0][i]
        item_dict["attention_mask"]=self.encoded_data[1][i]
        item_dict["labels"]=self.label_list[i]
        return item_dict

train_df = pd.read_csv("train.csv")
#train_data
val_df = pd.read_csv("val.csv")  
#val_data
test_df = pd.read_csv("test.csv") 
#test_data

# Initialize Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("google/canine-s")
# deberta_model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

train_data = DebertaDataset(train_df, tokenizer)
val_data = DebertaDataset(val_df, tokenizer)
test_data = DebertaDataset(test_df, tokenizer)

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## Return a dictionary containing the accuracy, f1, precision, and recall scores.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    model = AutoModelForSequenceClassification.from_pretrained('google/canine-s')
    #model = model.to('cuda')
    return model

## hyperparameter search
"""Run a hyperparameter search on a DebertaV3 model fine-tuned on news."""



training_args = TrainingArguments(
    output_dir="./checkpoint",  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    learning_rate=2e-5,  # Initial learning rate for the optimizer
    evaluation_strategy="epoch",
)

## TODO: Initialize a transformers.Trainer object and run a Bayesian
## hyperparameter search for at least 5 trials (but not too many) on the 
## learning rate. Use the hp_space parameter in hyperparameter_search() to specify
## your hyperparameter search space. (Note that this parameter takes a function
## as its value.)

tune_config = {
    "learning_rate": tune.uniform(1e-5, 5e-5),
}

trainer = Trainer(
    model_init=model_init,
    args=training_args,  # training arguments, defined above
    train_dataset=train_data,  # training dataset
    eval_dataset=val_data,  # evaluation dataset
    compute_metrics=compute_metrics,
)
# trainer.train()


best_run = trainer.hyperparameter_search(
    hp_space=lambda _: tune_config,
    backend="ray",
    n_trials=5,
    search_alg=BayesOptSearch(),
    mode="min",
    compute_objective=lambda metrics: metrics["eval_loss"],
    local_dir="./checkpoint",
    name="tune_transformer_bayes",
    log_to_file=True,
    # resources_per_trial={"cpu": 1, "gpu": 1}
)



print(f"Best run ID: {best_run.run_id}")
print(f"Objective: {best_run.objective}")
print(best_run.hyperparameters)
