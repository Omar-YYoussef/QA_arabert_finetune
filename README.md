# Arabic Question Answering System using AraBART

## Overview
This project implements an Arabic Question Answering (QA) system using the AraBART model fine-tuned on the ARCD (Arabic Reading Comprehension Dataset). The system can extract answers from given Arabic text contexts based on questions posed in Arabic.

## Project Structure
```
Omar-YYoussef-QA_arabert_finetune/
├── QA_arabert_finetune_code.ipynb    # Training notebook
├── main.py                           # Main application script
└── model.py                          # Model implementation
```

## Features
- Fine-tuned AraBART model for Arabic question answering
- Support for context-based question answering
- Easy-to-use pipeline interface
- Jupyter notebook for model training
- Modular code structure

## Installation

### Prerequisites
- Python 3.10 or higher
- PyTorch
- Transformers
- Datasets
- NumPy
- Pandas

### Setup
1. Clone the repository:
```bash
git clone [repository-url]
cd Omar-YYoussef-QA_arabert_finetune
```

2. Install required packages:
```bash
pip install torch transformers datasets numpy pandas
```

## Usage

### Using the Pre-trained Model
```python
from model import answer_question, qa_pipeline

# Define your context and question in Arabic
context = "Your Arabic context here"
question = "Your Arabic question here"

# Get the answer
answer = answer_question(qa_pipeline, context, question)
print(f"Answer: {answer}")
```

### Training the Model
The training process is documented in `QA_arabert_finetune_code.ipynb`. The notebook includes:
- Data preprocessing
- Model configuration
- Training pipeline setup
- Training execution
- Model evaluation

## Model Details

### Architecture
- Base Model: BERT-base-arabert
- Fine-tuned on: ARCD dataset
- Task: Extractive Question Answering

### Training Parameters
- Learning Rate: 2e-4
- Batch Size: 8
- Epochs: 3
- Optimizer: AdamW
- Weight Decay: 0.02
- Warmup Ratio: 0.1

## Files Description

### QA_arabert_finetune_code.ipynb
- Contains the complete training pipeline
- Includes data preprocessing steps
- Model training configuration
- Training execution code
- Evaluation metrics

### main.py
- Entry point for using the trained model
- Provides simple interface for question answering
- Example usage implementation

### model.py
- Model loading and initialization
- Question answering pipeline implementation
- Utility functions for text processing

## Performance
The model is trained on the ARCD dataset with the following metrics:
- Training Loss: Converges to ~0.95
- Validation Loss: ~2.8

## Limitations
- Works best with Modern Standard Arabic (MSA)
- Context length is limited to model's maximum sequence length
- Requires well-formed Arabic text input

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.



## Acknowledgments
- AraBART model developers
- ARCD dataset creators
- Hugging Face Transformers library
