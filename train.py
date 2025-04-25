import logging
import torch
from models.QISNet import QISNet, QISNet_Trainer
import utils.utils as utils


# TODO: tensorboard

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

if __name__ == '__main__':
    trainer = None

    model_name = 'QISNet'

    save_path = f"results/{model_name}_train/{utils.date_fname()}"
    utils.safe_mkdirs(save_path, log)

    T = 50  # num of bit-plane images
    alpha = 1.0  # gain
    epsilon = 1e-6

    num_epochs = 10
    lr = 1e-4
    criterion = torch.nn.MSELoss()
    batch_size = 4
    train_rate = 0.8
    shuffle = True
    drop_last = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'QISNet':
        model = QISNet(
            T,
            alpha
        )
        trainer = QISNet_Trainer(
            model,
            num_epochs,
            lr,
            criterion,
            T,
            alpha,
            batch_size,
            train_rate,
            shuffle,
            drop_last,
            save_path,
            device,
        )

    trainer.train()
