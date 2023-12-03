# Project-2-Text-Summarization

This project aims to develop a text summarization system using the cahya/bert-base-indonesian-522M Transformer model on a dataset collected from Liputan 6 news sources. Text summarization technology can process and summarize long texts automatically, so it can help users to get important information quickly and efficiently.

**Paper reference:** [Liputan6: A Large-scale Indonesian Dataset for Text Summarization](https://aclanthology.org/2020.aacl-main.60.pdf) AACL-IJCNLP 2020)

## Instalation
**Install wiht pip**

Install the transformers, datasets, evaluate, and rouge_score with pip:

```
!pip install transformers datasets evaluate rouge_score
```
Ugrade tensorflow with pip:

```
!pip install --upgrade tensorflow
```
Install accelerate version 0.20.1

```
!pip install accelerate>=0.20.1
```

Install rouge with pip:

```
! pip install rouge
```

**PyTorch with CUDA**

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow [PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.

## Dataset

In this experiment, we use the [Liputan-6](https://drive.google.com/drive/folders/1YZs2pwCNmRg9PezbKDodYCX9LlGTifu8)
 dataset

 ## Preprocess

 The next step is to load a cahya/bert-base-indonesian-1.5G to process text and summary with _GPT2Tokenizer, and BertTokenizer_ :

```
from transformers import GPT2Tokenizer, BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('cahya/bert-base-indonesian-1.5G')

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('cahya/gpt2-small-indonesian-522M')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
model.config.eos_token_id = gpt2_tokenizer.eos_token_id
model.config.pad_token_id = gpt2_tokenizer.pad_token_id
```

## Evaluate

For this task, load the ROUGE metric (see the  Evaluate quick tour to learn more about how to load and compute a metric):

```
import evaluate

rouge = evaluate.load("rouge")
```

Then create a function that passes your predictions and labels to compute to calculate the ROUGE metric:

```
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = bert_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, bert_tokenizer.pad_token_id)

    # Convert labels from token IDs to text tokens
    decoded_labels = []
    for label_ids in labels:
        decoded_label = bert_tokenizer.convert_ids_to_tokens(label_ids)
        decoded_label = " ".join(decoded_label)
        decoded_labels.append(decoded_label)

    # Process decoded predictions and labels
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Compute ROUGE metrics
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    return {key: value * 100 for key, value in result.items()}
```

## Train

We ready to start training model now! Load bert-base-indonesian-522M and cahya/gpt2-small-indonesian-522M  with EncoderDecoderModel:

```
from transformers import BertConfig, GPT2Config, EncoderDecoderModel, BertModel, GPT2LMHeadModel

#load the pre_trained models
encoder_config = BertConfig.from_pretrained('cahya/bert-base-indonesian-522M')
decoder_config = GPT2Config.from_pretrained('cahya/gpt2-small-indonesian-522M')

#Make sure that the decoder confiq is set as a decoder
decoder_config.is_decoder = True
decoder_config.add_cross_attention = True

#Create the BERT encoder and GPT-2 decoer with the
encoder = BertModel.from_pretrained('cahya/bert-base-indonesian-1.5G', config=encoder_config)
decoder = GPT2LMHeadModel.from_pretrained('cahya/gpt2-small-indonesian-522M', config=decoder_config)

#Combine the encoder and decoder into as Seq2Seq model
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

#Set the decoder_start_token_id
model.config.decoder_start_token_id=3

#Set the other generation parameters
model.config.eos_token_id=1
model.config.pad_token_id=2
model.config.max_lenght=40
model.config.min_lenght=20
model.config.no_repeat_gram_size=3
model.config.early_stopping=True
model.config.lenght_penalty=2.0
model.config.num_beams=10
gradient_checkpointing=True
```

At this point, only three steps remain:

1. Define your training hyperparameters in **Seq2SeqTrainingArguments**. The only required parameter is output_dir which specifies where to save your model. You’ll push this model to the Hub by setting push_to_hub=True (you need to be signed in to Hugging Face to upload your model). At the end of each epoch, the Trainer will evaluate the **ROUGE** metric and save the training checkpoint.
2. Pass the training arguments to **Seq2SeqTrainer** along with the model, dataset, tokenizer, data collator, and compute_metrics function.
3. Call **train()** to finetune your model.

```
from transformers import  Seq2SeqTrainingArguments, Seq2SeqTrainer

import os
os.environ["WANDB_DISABLED"] = "true"

training_args = Seq2SeqTrainingArguments(
    output_dir="/content/liputan_6_model",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    logging_steps=10,
    save_steps=16,
    # num_train_epochs=4,
    warmup_steps=1,
    max_steps=16,
    save_total_limit=1,
    weight_decay=0.01,
    # fp16=True
)

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    data_collator=data_collator,
    tokenizer=bert_tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
```

## Result

<img width="667" alt="image" src="https://github.com/verdwis/Project-2-Text-Summarization/assets/101826376/510aae53-cf79-4223-8198-fe5079044e6d">






 

 
