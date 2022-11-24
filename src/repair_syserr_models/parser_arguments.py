#!/usr/bin/env python3

import argparse


def getArgs(argv=None):
    parser = argparse.ArgumentParser(
        description="systematic-error-gen-models",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="batch size for training",
    )

    parser.add_argument(
        "--number-epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to run for training",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="lr",
        help="initial learning rate for optimizer",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--cuda-on",
        action="store_true",
        default=False,
        help="Use CUDA (GPU) to run experiment",
    )

    parser.add_argument(
        "--dataset-folder",
        default="standard",
        type=str,
        dest="data_folder",
        help="Input dataset folder to use in experiments",
    )

    parser.add_argument(
        "--save-on",
        action="store_true",
        default=False,
        help="True / False on saving experiment data",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="VAE",
        help="name of the file with the generative model",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default="./dummy/",
        help="output folder path where experiment data is saved",
    )

    parser.add_argument(
        "--embedding-size",
        type=int,
        default=50,
        metavar="N",
        help="size of the embeddings for the categorical attributes",
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=15,
        metavar="N",
        help="dimension of the latent space",
    )

    parser.add_argument(
        "--layer-size",
        type=int,
        default=400,
        metavar="N",
        help="capacity of the internal layers of the models",
    )

    parser.add_argument(
        "--verbose-metrics-epoch",
        action="store_true",
        default=False,
        help="show / no show the metrics for each epoch",
    )

    parser.add_argument(
        "--verbose-metrics-feature-epoch",
        action="store_true",
        default=False,
        help="show / no show the metrics for each epoch -- feature stuff",
    )

    parser.add_argument(
        "--l2-reg",
        default=0.0,
        type=float,
        help="e.g. values lie between 0.1 and 100. if high corruption, default turned off",
    )

    parser.add_argument(
        "--activation",
        default="relu",
        type=str,
        help="either choose ''relu'' or ''hardtanh'' (computationally cheaper than tanh)",
    )

    parser.add_argument(
        "--semi-supervise",
        action="store_true",
        default=False,
        help="use trusted set for additional training",
    )  # TODO: make model dependent? (some models are semi-supervised only)

    parser.add_argument(
        "--sup-loss-coeff",
        default=0.1,
        type=float,
        help="coefficient for supervised loss (e.g. cross-entropy with labelled data)",
    )

    ## Additions to basic below:

    # semi_y_VAE_GMM (VAE GMM in paper): 2-comp GMM p(z|y) model options
    parser.add_argument(
        "--learn-z-given-y-priors",
        action="store_true",
        default=False,
        help="2 comp z|y model: should the priors mean and variance be learnt",
    )
    # semi_y_VAE_GMM (VAE GMM in paper): alternative to above option, fix the prior variances for p(z|y) model
    parser.add_argument("--fixed-prior-zy0-sigma", default=5.0, type=float)
    parser.add_argument("--fixed-prior-zy1-sigma", default=1.0, type=float)

    # static prior on y (ratio of clean data points), used in several models (e.g. semi_y_CLSVAE, semi_y_VAE_GMM (VAE GMM), semi_y_CCVAE)
    parser.add_argument("--y-clean-prior", default=0.7, type=float)

    # KL weighting and annealing
    parser.add_argument(
        "--kl-anneal",
        action="store_true",
        default=False,
        help="run VAE model with or without KL annealing",
    )
    parser.add_argument("--kl-beta-const", default=1.0, type=float)

    parser.add_argument("--kl-anneal-cycles", default=1, type=int)
    parser.add_argument("--kl-anneal-ratio", default=0.5, type=float)
    parser.add_argument("--kl-anneal-delay-epochs", default=5, type=int)

    parser.add_argument("--kl-anneal-start", default=1e-6, type=float)
    parser.add_argument("--kl-anneal-stop", default=1.0, type=float)

    ## load model flags
    parser.add_argument("--load-model", action="store_true", default=False)
    parser.add_argument("--load-model-path", type=str)

    # Image Dataset: i.e. using BCE_loss for pixels (binary images), and data loading.
    parser.add_argument("--use-binary-img", action="store_true", default=False)

    parser.add_argument(
        "--use-batch-norm", action="store_true", default=False
    )  # use_batch_norm between layers of encoder / decoder NNs

    # controls z_\epsilon noise standard deviation
    parser.add_argument(
        "--sigma-eps-z-in",
        default=0.1,
        type=float,
        help="sigma of eps noise for z_d y=1 (inlier) representation",
    )

    # controls z_\epsilon noise mean (we use 0 throughout)
    parser.add_argument(
        "--mean-eps-z-in",
        default=0.0,
        type=float,
        help="mean of eps noise for z_d y=1 (inlier) representation",
    )

    ## use the weighted cross-entropy for labelled data loss 
    # (weighting uses number of inlier and outliers, see CLSVAE paper)
    parser.add_argument("--use-sup-weights", action="store_true", default=False)

    # name of file for trusted-set (see gen_utils.py for format; usually same folder as other data)
    parser.add_argument("--trust-set-name", default=None, type=str)

    ## semi_y_CLSVAE model: prior structure for partitioned latent space options
    # z_clean, or z_c (as in CLSVAE paper)
    # z_dirty, or z_d (as in CLSVAE paper)
    parser.add_argument(
        "--fixed-prior-z-clean",
        default=1.0,
        type=float,
        help="controls the standard deviation of z_c (or z_clean)",
    )
    parser.add_argument(
        "--fixed-prior-z-dirty",
        default=5.0,
        type=float,
        help="controls the standard deviation of z_d (or z_dirty)",
    )

    ## semi_y_CLSVAE: regularizer for minizing mutual information
    parser.add_argument("--dist-corr-reg", action="store_true", default=False)
    parser.add_argument("--dist-corr-reg-coeff", default=1.0, type=float)

    parser.add_argument("--reg-delay-n-epochs", default=5, type=int)
    parser.add_argument("--reg-schedule-ratio", default=0.5, type=float)

    # full_y_CVAE (supervised) options (test type of encoder)
    parser.add_argument("--use-q-z-y", action="store_true", default=False)

    # semi_y_CCVAE (semi-supervised) options
    parser.add_argument(
        "--q-y-x-coeff",
        default=1.0,
        type=float,
        help="semi-supervision coefficient for labelled data loss, used in CCVAE model",
    )  # 10 / 100 / 1000

    # testing option (train on clean data instead -- was useful to test compression hypothesis on standard VAE)
    parser.add_argument("--train-on-clean-data", action="store_true", default=False)

    return parser.parse_args(argv)
