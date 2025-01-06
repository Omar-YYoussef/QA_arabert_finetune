Omar-YYoussef-QA_arabert_finetune

This project fine-tunes the AraBERT model for Question Answering on the ARCD dataset. It includes code for data preprocessing, model training, and inference.

Project Structure
Omar-YYoussef-QA_arabert_finetune/
├── QA_arabert_finetune_code.ipynb
├── main.py
└── model.py
content_copy
download
Use code with caution.

QA_arabert_finetune_code.ipynb: This Jupyter Notebook contains the complete code for data loading, preprocessing, model fine-tuning, and evaluation. It uses the datasets and transformers libraries.

main.py: This script demonstrates how to use the fine-tuned model for answering questions. It imports the necessary functions from model.py.

model.py: This module contains functions for loading the pre-trained model and tokenizer, and also provides a function to answer questions using the pipeline.

Data

The project uses the ARCD dataset, which is automatically downloaded and processed using the datasets library. The dataset is preprocessed and tokenized using the tokenizer from the aubmindlab/bert-base-arabert model.

Dependencies

The following Python libraries are required to run this project:

numpy

pandas

torch

transformers

datasets

You can install these libraries using pip:

pip install numpy pandas torch transformers datasets
content_copy
download
Use code with caution.
Bash
Usage
1. Fine-tuning the Model

Open QA_arabert_finetune_code.ipynb in a Jupyter Notebook environment (e.g., Kaggle, Colab, or locally with Jupyter).

Execute the cells sequentially. This will:

Load and preprocess the ARCD dataset.

Load the AraBERT model and tokenizer.

Fine-tune the model on the training dataset.

Evaluate the model on the validation dataset.

Save the trained model and tokenizer.

Push the model to the hub in this repo "gp-tar4/QA_FineTuned_Arabert".

Note: the model will be pushed to hugging face hub so you must have an account with a user name specified in the notebook

2. Using the Fine-tuned Model for Question Answering

Ensure that the fine-tuned model and tokenizer are saved and accessible (the notebook saves them).

Modify the context and question variables in main.py with your desired text.

Run the main.py script:

python main.py
content_copy
download
Use code with caution.
Bash

This will:

Load the fine tuned model and tokenizer from the hub

Print the answer to the provided question based on the context.

Model details

The model is aubmindlab/bert-base-arabert after finetuning on the ARCD dataset

Code Explanation
QA_arabert_finetune_code.ipynb

Data Loading and Preprocessing: The notebook loads the ARCD dataset, preprocesses it by removing unnecessary columns and preparing it for question answering. The preprocess_function function tokenizes the input text and sets the start and end positions for the answers.

Model Fine-tuning: The notebook sets up the training parameters, loads the pre-trained AraBERT model, and fine-tunes it on the preprocessed dataset using the Trainer class from the transformers library.

model.py

load_qa_model(model_name): This function loads the specified pre-trained model and tokenizer and creates a question-answering pipeline.

answer_question(pipeline, context, question): This function takes a question and a context and returns the predicted answer.

qa_pipeline: This variable loads the trained model pipeline when the model.py module is imported.

main.py

This script imports the answer_question function and the loaded qa_pipeline from model.py.

It sets the context and question variables.

It calls answer_question to get the answer and print it to the console.

Note

This project uses Weights & Biases (wandb) for logging. You'll need a wandb account and API key.

The model will be pushed to your account on huggingface hub.

You may need to modify the TrainingArguments in QA_arabert_finetune_code.ipynb based on your computational resources.

Evaluation Metrics

The following table shows the evaluation metrics for three models: ArabianGpt01, AraBert, and DistilBert.

Models	ROUGE-1	ROUGE-2	ROUGE-L	WER	BLEU
ArabianGpt01	0.8464	0.7562	0.8443	0.3190	0.6353
AraBert	0.8415	0.7537	0.8391	0.2765	0.6101
DistilBert	0.2625	0.1117	0.2492	1.0226	0.0240
