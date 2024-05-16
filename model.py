from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering


# Function to load the model and tokenizer
def load_qa_model(model_name):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return qa_pipeline


# Function to perform question answering
def answer_question(pipeline, context, question):
    result = pipeline(question=question, context=context)
    return result['answer']


# Load the model and tokenizer
qa_pipeline = load_qa_model('gp-tar4/QA_FineTuned_Arabert')

# Export the pipeline for use in another file
__all__ = ['answer_question', 'qa_pipeline']
