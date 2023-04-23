import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoModelForQuestionAnswering
from models import BaselineModel
from utils.options import train_options
from utils.set_seed import seed_everything


def create_prediction(test_dataframe, qa_pipepline, path_to_save='predictions.json'):
    extracted_part = []
    for i in tqdm(range(len(test_dataframe))):
        row = test_dataframe.iloc[i]
        question, context = row['label'], row['text']
        prediction = qa_pipepline(question=question, context=context)
        row_extracted_part = {
            'text': [prediction['answer']],
            'answer_start': [prediction['start']],
            'answer_end': [prediction['end']]
        }
        extracted_part.append(row_extracted_part)
        
    test_dataframe['extracted_part'] = extracted_part
    test_dataframe.to_json(path_to_save)


if __name__ == '__main__':
    arguments = train_options()  
    seed_everything(arguments.seed)
    test_dataframe = pd.read_json(arguments.test_file_path)
    
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
        create_prediction(test_dataframe, qa_pipepline)
