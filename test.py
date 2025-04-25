import logging
import torch
import utils.utils as utils
from dataloaders.get_dataloaders import get_test_dataloaders
from models.DU_ISTA import DU_ISTA, DU_ISTA_Tester
from models.QISNet import QISNet, QISNet_Tester
from models.TV_DU_ADMM import TV_DU_ADMM, TV_DU_ADMM_Tester
from models.MLE import MLE, MLE_Tester
from models.TV_ADMM import TV_ADMM, TV_ADMM_Tester
from models.PnP_ADMM_BM3D import PnP_ADMM_BM3D, PnP_ADMM_BM3D_Tester
from models.PnP_ADMM_DRUNet import PnP_ADMM_DRUNet, PnP_ADMM_DRUNet_Tester
from models.PnP_ADMM_DnCNN import PnP_ADMM_DnCNN, PnP_ADMM_DnCNN_Tester
from models.TV_DU_ADMM_LK import TV_DU_ADMM_LK, TV_DU_ADMM_LK_Tester

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger(__name__)

    tester = None

    ## proposed
    # model_name = 'TV_DU_ADMM_LK'
    # model_name = 'TV_DU_ADMM'

    ## model-based
    # model_name = 'MLE'
    # model_name = 'TV_ADMM'

    ## PnP-based
    model_name = 'PnP_ADMM_BM3D'
    # model_name = 'PnP_ADMM_DnCNN'
    # model_name = 'PnP_ADMM_DRUNet'

    ## deep unfolding-based
    # model_name = 'DU_ISTA'

    ## deep-based
    # model_name = 'QISNet'

    save_path = f"results/{model_name}/{utils.date_fname()}"
    utils.safe_mkdirs(save_path, log)

    T = 32  # num of bit-plane images
    alpha = 1.0  # gain
    epsilon = 1e-6

    test_loader = get_test_dataloaders(T)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'TV_DU_ADMM_LK':
        dname = '2025-02-19_23.36.38.034333'

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
            epsilon,
        )

        trained_weights_path = f"results/{model_name}_train/{dname}/model.pt"
        state_dict = torch.load(trained_weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        tester = TV_DU_ADMM_LK_Tester(model, test_loader, save_path, device)
    elif model_name == 'TV_DU_ADMM':
        dname = '2025-02-16_04.41.11.204253'

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
            epsilon,
        )

        trained_weights_path = f"results/{model_name}_train/{dname}/model.pt"
        state_dict = torch.load(trained_weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        tester = TV_DU_ADMM_Tester(model, test_loader, save_path, device)
    elif model_name == 'MLE':
        model = MLE(
            T,
            alpha
        )
        tester = MLE_Tester(model, test_loader, save_path, device)
    elif model_name == 'TV_ADMM':
        num_itr = 30
        lambda_ = 0.1  # weight of regularized term
        eta = 0.15  # step size for updating x
        rho = 0.3  # initial weight for penalty term

        model = TV_ADMM(
            T,
            alpha,
            num_itr,
            lambda_,
            eta,
            rho,
            epsilon,
        )
        tester = TV_ADMM_Tester(model, test_loader, save_path, device)
    elif 'PnP_ADMM' in model_name:
        num_itr = 30
        lambda_ = 0.1  # weight of regularized term
        eta = 0.1  # step size for updating x
        rho_init = 0.1  # initial weight for penalty term
        beta = 0.5  # parameter for difference comparison
        gamma = 1.2  # step size for updating rho
        rho_update_differential = True  # True: using difference comparison, False: monotonically increasing

        if model_name == 'PnP_ADMM_BM3D':
            model = PnP_ADMM_BM3D(
                T,
                alpha,
                num_itr,
                lambda_,
                eta,
                rho_init,
                beta,
                gamma,
                rho_update_differential,
                epsilon,
            )
            tester = PnP_ADMM_BM3D_Tester(model, test_loader, save_path, device)
        elif model_name == 'PnP_ADMM_DnCNN':
            model = PnP_ADMM_DnCNN(
                T,
                alpha,
                num_itr,
                lambda_,
                eta,
                rho_init,
                beta,
                gamma,
                rho_update_differential,
                epsilon,
            )
            tester = PnP_ADMM_DnCNN_Tester(model, test_loader, save_path, device)
        elif model_name == 'PnP_ADMM_DRUNet':
            model = PnP_ADMM_DRUNet(
                T,
                alpha,
                num_itr,
                lambda_,
                eta,
                rho_init,
                beta,
                gamma,
                rho_update_differential,
                epsilon,
            )
            tester = PnP_ADMM_DRUNet_Tester(model, test_loader, save_path, device)
    elif model_name == 'DU_ISTA':
        dname = '2025-02-16_13.37.24.266476'

        num_itr = 30
        mu = 0.1  # weight of regularized term
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
        trained_weights_path = f"results/{model_name}_train/{dname}/model.pt"
        state_dict = torch.load(trained_weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        tester = DU_ISTA_Tester(model, test_loader, save_path, device)
    elif model_name == 'QISNet':
        dname = '2025-02-11_03.06.04.522025'
        model = QISNet(
            T,
            alpha
        )
        trained_weights_path = f"results/{model_name}_train/{dname}/model.pt"
        state_dict = torch.load(trained_weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        tester = QISNet_Tester(model, test_loader, save_path, device)

    tester.test()
