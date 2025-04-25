import logging
import torch
from models.DU_ISTA import DU_ISTA, DU_ISTA_Trainer
from models.QISNet import QISNet, QISNet_Trainer
from models.TV_DU_ADMM import TV_DU_ADMM, TV_DU_ADMM_Trainer
import utils.utils as utils
from models.TV_DU_ADMM_LK import TV_DU_ADMM_LK, TV_DU_ADMM_LK_Trainer

# TODO: tensorboard

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

if __name__ == '__main__':
    trainer = None

    # model_name = 'TV_DU_ADMM_LK'
    model_name = 'TV_DU_ADMM'
    # model_name = 'DU_ISTA'
    # model_name = 'QISNet'

    save_path = f"results/{model_name}_train/{utils.date_fname()}"
    utils.safe_mkdirs(save_path, log)

    T = 32  # num of bit-plane images
    alpha = 1.0  # gain
    epsilon = 1e-6

    num_epochs = 100
    lr = 1e-4
    criterion = torch.nn.MSELoss()
    batch_size = 64
    train_rate = 0.8
    shuffle = True
    drop_last = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'TV_DU_ADMM_LK':
        num_itr = 30
        lambda_init = 0.1  # weight of regularized term
        eta_init = 0.15  # step size for updating x
        rho_init = 0.3  # initial weight for penalty term

        model = TV_DU_ADMM_LK(
            T,
            alpha,
            num_itr,
            lambda_init,
            eta_init,
            rho_init,
            epsilon
        )
        trainer = TV_DU_ADMM_LK_Trainer(
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
    elif model_name == 'TV_DU_ADMM':
        num_itr = 30
        lambda_init = 0.1  # weight of regularized term
        eta_init = 0.15  # step size for updating x
        rho_init = 0.3  # initial weight for penalty term

        model = TV_DU_ADMM(
            T,
            alpha,
            num_itr,
            lambda_init,
            eta_init,
            rho_init,
            epsilon
        )
        trainer = TV_DU_ADMM_Trainer(
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
    elif model_name == 'DU_ISTA':
        num_itr = 30
        mu = 0.1
        eta = 0.05
        D_init = torch.load("chkpt/dictionary.pt", weights_only=True)
        c = 0.1

        model = DU_ISTA(
            T,
            alpha,
            num_itr,
            mu,
            eta,
            D_init,
            c,
            epsilon
        )
        trainer = DU_ISTA_Trainer(
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
    elif model_name == 'QISNet':
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
