import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, default_data_collator
from torch.utils.data import DataLoader
from utils.options import train_options
from utils.dataset import DocumentDataset
from utils.set_seed import seed_everything


def get_logits(model, test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = DataLoader(dataset=test_dataset, batch_size=16,
                             shuffle=False, collate_fn=default_data_collator,
                             num_workers=0)
    
    model.eval()
    start_logits = []
    end_logits = []
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(start_logits)[: len(test_dataset)]
    end_logits = np.concatenate(end_logits)[: len(test_dataset)]
    return start_logits, end_logits 


def get_prediction(start_logits, end_logits, dataset, n_best: int = 20, max_answer_length: int = 310):
    predicted_answers = []
    for idx, example in tqdm(enumerate(dataset)):
        example_id = example['id']
        context = dataset.get_context(example_id.item())
        answers = []

        start_logit = start_logits[idx]
        end_logit = end_logits[idx]
        offsets = example['offset_mapping']

        start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
        end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()

        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip answers with a length that is either < 0 or > max_answer_length
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue

                answers.append(
                    {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                        "answer_start": offsets[start_index][0].item(),
                        "answer_end": offsets[end_index][1].item()
                    }
                )

        best_answer = max(answers, key=lambda x: x["logit_score"])
        
        predicted_answers.append(
            {
                "text": [best_answer["text"]],
                "answer_start": [best_answer['answer_start']],
                "answer_end": [best_answer['answer_end']]
            }
        )
    return predicted_answers


if __name__ == '__main__':
    arguments = train_options()  
    seed_everything(arguments.seed)
    test_dataframe = pd.read_json(arguments.test_file_path)
    test_dataset = DocumentDataset(test_dataframe, with_answers=False)

    model = AutoModelForQuestionAnswering.from_pretrained(arguments.model_pretrained_path)
    start_logits, end_logits = get_logits(model, test_dataset)

    prediction = get_prediction(start_logits, end_logits, test_dataset)
    
    test_dataframe['extracted_part'] = prediction
    test_dataframe.to_json('predictions.json')