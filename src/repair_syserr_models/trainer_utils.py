# import torch
# import torch.nn.functional as F
from torch import optim


class StandardTrainer:
    def __init__(self, model, args, lr_opt=1e-3, weight_decay_opt=0.0):

        self.model = model
        self.args = args

        self.optimizer = optim.Adam(
            params=[p for p in self.model.parameters() if p.requires_grad],
            lr=lr_opt,
            weight_decay=weight_decay_opt,
        )

    def train_step(
        self, x_input, y_input, epoch, kl_beta, mask_ssup, reg_scheduler_val
    ):

        self.optimizer.zero_grad()

        p_params, q_params, q_samples, log_q_dists = self.model(
            x_input, n_epoch=epoch - 1, y_targets=y_input
        )

        loss_dict = self.model.loss_function(
            x_input,
            p_params,
            q_params,
            q_samples,
            log_q_dists,
            mask_ssup,
            y_input,
            sup_coeff=self.args.sup_loss_coeff,
            kl_coeff=kl_beta,
            reg_scheduler_val=reg_scheduler_val,
            n_epoch=epoch,
        )

        loss_dict["total_loss"].backward()

        self.optimizer.step()

        return loss_dict
