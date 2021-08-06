"""Conduct end to end training."""
import numpy as np
import tensorflow as tf

from data import Data
from sdf import SDFModel
from conditional import ConditionalModel
from loss import PricingLoss
from training import Trainer


def train_GAN(
    modified_training=True,
    normalize=False
):
    """Trains the GAN model."""
    ###############
    # Import data #
    ###############
    data_path = "../../../datasets"

    macro_path = f"{data_path}/macro/macro_train.npz"
    macro_zip = np.load(macro_path)

    firm_path = f"{data_path}/char/Char_train.npz"
    firm_zip = np.load(firm_path)

    ###################
    # Data processing #
    ###################
    cleaner = Data()
    inputs = cleaner.clean(macro_zip["data"],
                           firm_zip["data"])
    sample = [
        inputs[0],
        inputs[1][:, :5, :],
        inputs[2][:, :5],
        inputs[3][:, :5]
    ]

    #################
    # Create models #
    #################
    sdf_train = SDFModel()
    sdf_train(sample)

    conditional_train = ConditionalModel()
    conditional_train(sample)

    ######################
    # Loss and Optimizer #
    ######################
    optimizer = tf.keras.optimizers.Adam()
    loss = PricingLoss()

    ############
    # Training #
    ############
    trainer = Trainer(loss)
    if not modified_training:
        trainer.train(
            sdf_model=sdf_train,
            conditional_model=conditional_train,
            optimizer=optimizer,
            inputs=inputs,
            sdf_epoches_unc=10,
            moment_epoches=10,
            sdf_epoches_cond=10,
            steps=100,
            normalize=normalize
        )
    if modified_training:
        trainer.modified_train(
            sdf_model=sdf_train,
            conditional_model=conditional_train,
            optimizer=optimizer,
            inputs=inputs,
            sdf_epoches_unc=100,
            epoches=50,
            steps=100,
            normalize=normalize
        )

    return sdf_train, conditional_train


if __name__ == "__main__":
    sdf_train, conditional_train = train_GAN(
        modified_training=True,
        normalize=False)
