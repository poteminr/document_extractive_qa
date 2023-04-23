import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

    
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
    
    train_dataframe, val_dataframe = train_test_split(dataframe, test_size=test_size, random_state=1007)
    train_dataset = DocumentDataset(train_dataframe)
    val_dataset = DocumentDataset(val_dataframe)
    return train_dataset, val_dataset
    

class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, encoder_model: str = 'cointegrated/rubert-tiny2'):
        self.dataframe = dataframe
        self.encoder_model = encoder_model
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
        self.add_token_positions(tokenized_input, answers)
        tokenized_input.update({'id':ids})
        return tokenized_input
    
    def get_context(self, sample_id: int):
        return self.dataframe[self.dataframe.id == sample_id].text.iloc[0]
    
    def get_answer(self, sample_id: int):
        answer = self.dataframe[self.dataframe.id == sample_id].extracted_part.iloc[0]
        return {'text': answer['text'], 'answer_start':answer['answer_start']}
        