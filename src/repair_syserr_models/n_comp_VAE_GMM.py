import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict

from repair_syserr_models.loss_utils import n_comp_vae_gmm_ELBO
from repair_syserr_models.module_utils import baseEncoder, baseDecoder
from repair_syserr_models.module_utils import encodeMVNDiag, encodeCateg, GMMDistModule


EPS = 1e-8

## multiple component VAE-GMM model -- p(x|z) p(z|a) p(a)

# p(a) is categorical distribution, reflecting probability of component
# p(z|a) is Gaussian component
# p(x|z) is standard decoder


class VAE(nn.Module):

    """ 
        Unsupervised n-component VAE-GMM
    
        -- this version reconstruction term's expectation is enumerated (analytical)

        -- inspired by general VAE-GMM from Rui Shu et al. for semi-supervised VAEs

        -- this was implemented as a fully unsupervised method

    """

    def __init__(self, dataset_obj, args):

        super().__init__()
        # NOTE: for feat_select, (col_name, col_type, feat_size) in enumerate(dataset_obj.feat_info)

        self.dataset_obj = dataset_obj
        self.args = args

        self.args.size_input = len(
            dataset_obj.cat_cols
        ) * self.args.embedding_size + len(dataset_obj.num_cols)
        self.args.size_output = len(dataset_obj.cat_cols) + len(
            dataset_obj.num_cols
        )  # 2*

        if args.activation == "relu":
            self.activ = nn.ReLU()
        elif args.activation == "hardtanh":
            self.activ = nn.Hardtanh()

        # vae neural network architecture (encoder and decoder)
        self.latent_dim = args.latent_dim
        latent_dim = args.latent_dim
        layers_enc = [(None, 200), (200, 100), (100, 50)]  # encoder
        layers_dec = [(latent_dim, 50), (50, 100), (100, 200)]  # decoder

        # number of components for GMM in z-space, i.e. number of categories of p(a)
        self.n_comps = 10

        enc_args_a = {"layers_list": layers_enc, "activ": self.activ}
        self.var_a_encoder = encodeCateg(
            dataset_obj, args, self.n_comps, mod_torso_enc=None, **enc_args_a
        )

        enc_args_z = {
            "layers_list": layers_enc,
            "activ": self.activ,
            "add_in_len": self.n_comps,
        }
        self.var_z_encoder = encodeMVNDiag(
            dataset_obj, args, latent_dim, mod_torso_enc=None, **enc_args_z
        )

        # GMM prior: categ dist + gauss dist components
        _static_prior_categ = {"logits": torch.zeros(self.n_comps)} # uniform prior
        # torch.log(torch.ones(self.n_comps).float()/self.n_comps) or torch.zeros(self.n_comps) - uniform prior
        self.prior_gmm = GMMDistModule(
            self.n_comps, latent_dim, static_param_categ=_static_prior_categ
        )

        self.decoder = baseDecoder(dataset_obj, args, layers_dec, self.activ)

        # aux for storing info. later (total_loss is elbo)
        self.loss_ret_names = ["total_loss", "nll", "kld_z_tot", "kld_a"]

    def forward(self, x_data, n_epoch=None, y_targets=None):

        # p_params --> recon params (softmax probs and gaussian per feat.), and prior params
        # q_params --> mu, logvar of gauss variational dist
        # q_samples --> samples from gauss variational dist

        q_params = dict()
        q_samples = dict()
        log_q = dict()

        # a encoder: assignment of clusters (variational responsibilities)
        q_params["a"], q_samples["a"], log_q["a"] = self.var_a_encoder(
            x_data, sampling=False, evalprob=False
        )

        # z encoder -- used in KL computation
        a_comps_mask = (
            torch.eye(self.n_comps).to(x_data.device).repeat(x_data.shape[0], 1)
        )
        x_data_repeat = (
            x_data.unsqueeze(1).repeat(1, self.n_comps, 1).view(-1, x_data.shape[1])
        )

        q_params["z_a"], q_samples["z_a"], _ = self.var_z_encoder(
            x_data_repeat, add_in=a_comps_mask, sampling=True, evalprob=False
        )

        # NOTE: [(B.K),D] --> [B,K,D], for enumeration type KL of GMM
        q_params["z_a"]["mu"] = q_params["z_a"]["mu"].view(
            -1, self.n_comps, self.latent_dim
        )
        q_params["z_a"]["logvar"] = q_params["z_a"]["logvar"].view(
            -1, self.n_comps, self.latent_dim
        )

        _q_samples_z_a = q_samples["z_a"]  # [(B.K),D]
        q_samples["z_a"] = q_samples["z_a"].view(
            -1, self.n_comps, self.latent_dim
        )  # -1

        p_params = dict()  # generative / prior params
        _params_aux, _, _ = self.prior_gmm(
            sampling=False, evalprob=False
        )

        p_params["z"] = dict()
        p_params["a"] = dict()

        p_params["z"]["mu"] = _params_aux["mu"]
        p_params["z"]["logvar"] = _params_aux["logvar"]
        p_params["a"]["logits"] = _params_aux["logits"]

        p_params["recon_a"] = self.decoder(_q_samples_z_a)  # [(B.K), x_data size]

        if not self.training:  
            # used for evaluation, here we use MAP estimate.

            var_prob_a = F.softmax(q_params["a"]["logits"], dim=-1)

            _index_a = var_prob_a.max(dim=-1, keepdim=True)[1]
            _map_mask_a = torch.zeros_like(var_prob_a).scatter_(-1, _index_a, 1.0)

            q_params["z"], q_samples["z"], _ = self.var_z_encoder(
                x_data, add_in=_map_mask_a, sampling=True, evalprob=False
            )

            p_params["recon"] = self.decoder(q_params["z"]['mu'])

        return p_params, q_params, q_samples, []

    def loss_function(
        self,
        input_data,
        p_params,
        q_params,
        q_samples,
        log_q_dists,
        mask_semi=None,
        y_targets=None,
        sup_coeff=1.0,
        kl_coeff=1.0,
    ):

        input_repeat = (
            input_data.unsqueeze(1)
            .repeat(1, self.n_comps, 1)
            .view(-1, input_data.shape[1])
        )

        elbo_loss, nll, kld_z_tot, kld_a = n_comp_vae_gmm_ELBO(
            self.dataset_obj,
            input_repeat,
            p_params,
            q_params,
            kl_coeff=kl_coeff,  # 0.001; 0.1 ; 1.0
            n_comps=self.n_comps,
            data_size=input_data.shape[0],
        )

        ret_dict = OrderedDict(
            [
                ("total_loss", elbo_loss),
                ("nll", nll),
                (
                    "kld_z_tot",
                    kld_z_tot,
                ),
                ("kld_a", kld_a),
            ]
        )

        return ret_dict
