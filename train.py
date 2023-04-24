import wandb
from transformers import AutoModelForQuestionAnswering
from utils.dataset import get_train_val_dataset
from utils.options import train_options
from utils.set_seed import seed_everything
from trainer import TrainerConfig, Trainer


def train_config_to_dict(train_config: TrainerConfig):
    return dict((name, getattr(train_config, name)) for name in dir(train_config) if not name.startswith('__'))


if __name__ == "__main__":
    arguments = train_options()  
    seed_everything(arguments.seed)
    train_dataset, val_dataset = get_train_val_dataset(
        file_path=arguments.file_path,
        test_size=arguments.test_size,
        max_instances=arguments.max_instances,
        train_label=arguments.train_label
        )
    
    model = AutoModelForQuestionAnswering.from_pretrained(arguments.encoder_model)
    if arguments.model_pretrained_path is not None:
        model.from_pretrained(arguments.model_pretrained_path)

    config = TrainerConfig()
    config.model_type = arguments.model_type
    config.train_label = arguments.train_label
    config.encoder_model = arguments.encoder_model
    config.seed = arguments.seed
    
    wandb.init(project="Kontur", config=train_config_to_dict(config))
    trainer = Trainer(model=model, config=config, train_dataset=train_dataset, val_dataset=val_dataset)
    trainer.train()
