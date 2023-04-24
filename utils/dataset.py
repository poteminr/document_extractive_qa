import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset
    
    
def get_train_val_dataset(file_path: str, test_size: float = 0.25, max_instances: int = -1, train_label: int = -1):
    if train_label == 1:
        label_text = 'обеспечение исполнения контракта'
    elif train_label == 2:
        label_text = 'обеспечение гарантийных обязательств'
    
    dataframe = pd.read_json(file_path)
    
    if train_label != -1:
        dataframe = dataframe[dataframe.label == label_text]
    
    if max_instances != -1 and len(dataframe) > max_instances:
        dataframe = dataframe.sample(n=max_instances, random_state=1007)
    
    if test_size == 0:
        train_dataset = DocumentDataset(dataframe)
        val_dataset = None
    else:
        train_dataframe, val_dataframe = train_test_split(dataframe, test_size=test_size, random_state=1007)
        train_dataset = DocumentDataset(train_dataframe)
        val_dataset = DocumentDataset(val_dataframe)
        
    return train_dataset, val_dataset


class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, encoder_model: str = 'cointegrated/rubert-tiny2', with_answers=True):
        self.dataframe = dataframe
        self.encoder_model = encoder_model
        self.with_answers = with_answers
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model)
        self.tokenized_input = self.build_dataset()
                
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.tokenized_input.items()}

    def __len__(self):
        return len(self.tokenized_input.input_ids)
    
    def read_data(self):
        ids = self.dataframe['id'].values
        contexts = self.dataframe['text'].values
        questions = self.dataframe['label'].values
        answers = None
        
        if self.with_answers:
            answers = self.dataframe['extracted_part'].values
            
        return ids, list(contexts), list(questions), answers

    def add_token_positions(self, tokenized_input, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            if answers[i]['answer_end'][0] == answers[i]['answer_start'][0]:
                # the answer is not in the context
                start_positions.append(0)
                end_positions.append(0)
            else:    
                start_positions.append(tokenized_input.char_to_token(i, answers[i]['answer_start'][0], sequence_index=1))
                end_positions.append(tokenized_input.char_to_token(i, answers[i]['answer_end'][0] - 1, sequence_index=1))
            # if None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length
        tokenized_input.update({'start_positions': start_positions, 'end_positions': end_positions})
        
    def build_dataset(self):
        ids, contexts, questions, answers = self.read_data()            
        tokenized_input = self.tokenizer(questions, contexts, truncation=True, padding=True, return_offsets_mapping=True)
        
        if self.with_answers:
            self.add_token_positions(tokenized_input, answers)
            
        tokenized_input.update({'id':ids})
        return tokenized_input
    
    def get_context(self, sample_id: int):
        return self.dataframe[self.dataframe.id == sample_id].text.iloc[0]
    
    def get_answer(self, sample_id: int):
        if self.with_answers:
            answer = self.dataframe[self.dataframe.id == sample_id].extracted_part.iloc[0]
            return {'text': answer['text'], 'answer_start':answer['answer_start']}


def _get_train_val_sberquad(test_size: float = 0.2):
    dataset = load_dataset('sberquad', split='validation').train_test_split(test_size=test_size)
    train, test = dataset['train'], dataset['test']   
    train_dataset, test_dataset = SberquadDataset(train), SberquadDataset(test)
    return train_dataset, test_dataset


class SberquadDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, encoder_model: str = 'cointegrated/rubert-tiny2'):
        self.dataset = dataset.filter(lambda x: x["answers"]['answer_start'] != [-1])
        self.encoder_model = encoder_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model)
        self.tokenized_input = self.build_dataset()
                
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.tokenized_input.items()}

    def __len__(self):
        return len(self.tokenized_input.input_ids)
    
    def add_token_positions(self, tokenized_input, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            answer_start = answers[i]['answer_start'][0]
            answer_end = answer_start + len(answers[i]['text'][0])
            if answer_start == answer_end:
                # the answer is not in the context
                start_positions.append(0)
                end_positions.append(0)
            else:    
                start_positions.append(tokenized_input.char_to_token(i, answer_start, sequence_index=1))
                end_positions.append(tokenized_input.char_to_token(i, answer_end - 1, sequence_index=1))
            # if None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length
        tokenized_input.update({'start_positions': start_positions, 'end_positions': end_positions})
        
    def build_dataset(self):
        ids, contexts, questions, answers = self.dataset['id'], self.dataset['context'], self.dataset['question'], self.dataset['answers']
        tokenized_input = self.tokenizer(questions, contexts, truncation=True, padding=True, return_offsets_mapping=True)
        self.add_token_positions(tokenized_input, answers)
        tokenized_input.update({'id':ids})
        return tokenized_input
    
    def get_context(self, sample_id: int):
        # return self.dataframe[self.dataframe.id == sample_id].text.iloc[0]
        return self.dataset.filter(lambda x: x["id"] == sample_id)['context'][0]
    
    def get_answer(self, sample_id: int):
        answer = self.dataset.filter(lambda x: x["id"] == sample_id)['answers'][0]
        return {'text': answer['text'], 'answer_start':answer['answer_start']}
