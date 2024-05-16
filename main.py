from model import answer_question, qa_pipeline


# Define the context and question in Arabic
context = ""#EnterYourContext
question = ""#EnterYourQuestion

# Use the imported function to answer the question
answer = answer_question(qa_pipeline, context, question)

print(f"Answer: {answer}")