import os
import numpy as np
import logging
import torch


class CheckpointSaver:
    def __init__(self, dirpath, model_type='huggingface', decreasing=False, top_n=1):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.model_type = model_type
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        
    def __call__(self, model, epoch, metric_val):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + f'_epoch{epoch}')
        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save: 
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}")
            self.best_metric_val = metric_val
            
            if self.model_type == 'huggingface':
                model.save_pretrained(model_path)
            else:
                model_path = model_path + '.pt'
                torch.save(model.state_dict(), model_path)
            
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths)>self.top_n and self.model_type!='huggingface':
            # cant delete folderes because of permitions  
            # causes memory problems, need to fix
            self.cleanup()
    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]