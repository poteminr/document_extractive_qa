import torch
from transformers import pipeline, AutoModelForQuestionAnswering
from models import BaselineModel
from utils.options import train_options



if __name__ == '__main__':
    arguments = train_options()  
    
    if arguments.model_type == 'huggingface':
        model = AutoModelForQuestionAnswering.from_pretrained(arguments.encoder_model)
    else:        
        model = BaselineModel(encoder_model=arguments.encoder_model)

    if arguments.model_pretrained_path is not None:
        if arguments.model_type == 'huggingface':
            model.from_pretrained(arguments.model_pretrained_path)
        else:
            model.load_state_dict(torch.load(arguments.model_pretrained_path))

    if arguments.model_type == 'huggingface':
        qa_pipepline = pipeline("question-answering", model=model, tokenizer=arguments.encoder_model)