import wandb
from models import BaselineModel
from utils.dataset import get_train_val_dataset
from utils.options import train_options
from trainer import TrainerConfig, Trainer


def train_config_to_dict(train_config: TrainerConfig):
    return dict((name, getattr(train_config, name)) for name in dir(train_config) if not name.startswith('__'))


if __name__ == "__main__":
    arguments = train_options()  
    train_dataset, val_dataset = get_train_val_dataset(
        file_path=arguments.file_path,
        test_size=arguments.test_size,
        max_instances=arguments.max_instances
        )
    
    model = BaselineModel()
    config = TrainerConfig()
    
    wandb.init(project="Kontur", config=train_config_to_dict(config))
    trainer = Trainer(model=model, config=config, train_dataset=train_dataset, val_dataset=val_dataset)
    trainer.train()
