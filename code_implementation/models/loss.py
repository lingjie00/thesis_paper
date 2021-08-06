"""Defines the loss function for GAN model."""
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class PricingLoss(keras.losses.Loss):
    """Defines the Pricing Loss for GAN model."""
    def __init__(self):
        """Init."""
        super().__init__()

    def __call__(self,
                 SDF: tf.Tensor,
                 moment: tf.Tensor,
                 return_data: tf.Tensor,
                 mask: tf.Tensor,
                 weighted_loss: bool = True):
        """Compute GAN training loss."""
        ###################
        # Preprocess data #
        ###################
        mask = tf.cast(mask, "float")
        Ti = tf.reduce_sum(mask, axis=0)

        ################
        # Compute Loss #
        ################
        masked_return = return_data * mask
        empirical_sum = masked_return * SDF * moment
        empirical_sum = tf.reduce_sum(empirical_sum, axis=1)
        empirical_mean = empirical_sum / Ti
        loss = tf.square(empirical_mean)
        if weighted_loss:
            loss_weight = tf.reduce_sum(mask, axis=0)
            loss_weight_max = tf.reduce_max(loss_weight)
            loss_weight_normalized = loss_weight / loss_weight_max
            loss *= loss_weight_normalized
        loss = tf.reduce_mean(loss)
        return loss

    def sharpe(self, portfolio: tf.Tensor):
        """Calculate the SHARPE based on a portfolio of returns."""
        if not (type(portfolio) == np.ndarray):
            portfolio = portfolio.numpy()
        return np.mean(portfolio / portfolio.std())

    def sharpe_loss(self, sdf: tf.Tensor = None,
                    sdf_weight: tf.Tensor = None,
                    return_data: tf.Tensor = None,
                    mask: tf.Tensor = None,
                    normalize: bool = False):
        """Calculate the SHARPE as a loss based on sdf."""
        if not normalize and isinstance(type(sdf), type(None)):
            # check if either raw sdf is given or calculation is given
            raise ValueError("require either sdf or calculation")
        elif normalize:
            # only calculate weighted return if normalizing sdf
            masked_return = return_data[mask]
            weighted_return = masked_return * sdf_weight
            weight_index = np.sum(mask, axis=1).cumsum()[:-1]
            weighted_return_lst = np.split(weighted_return, weight_index)
            sdf = []
            for item in weighted_return_lst:
                sdf.append([item.sum()])
            sdf = np.array(sdf) + 1
        portfolio = 1 - sdf
        return self.sharpe(portfolio)

    def residual_loss(self,
                      return_data: tf.Tensor,
                      mask: tf.Tensor,
                      sdf_weight: tf.Tensor):
        """Compute residual loss."""
        Ni = tf.reduce_sum(tf.cast(mask, "int32"), axis=1)  # length of time
        masked_return = tf.boolean_mask(return_data, mask=mask)
        masked_return_lst = tf.split(masked_return, num_or_size_splits=Ni)
        weight_lst = tf.split(sdf_weight, num_or_size_splits=Ni)
        residual_square_lst = []
        return_square_lst = []
        for returnT, weightT in zip(masked_return_lst, weight_lst):
            estimated_return = tf.reduce_sum(returnT * weightT) /\
                    tf.reduce_sum(weightT * weightT) * weightT
            residual_return = tf.reduce_mean(
                tf.square(returnT - estimated_return)
            )
            residual_square_lst.append(residual_return)
            return_square = tf.square(returnT)
            return_square = tf.reduce_mean(return_square)
            return_square_lst.append(return_square)
        r2 = tf.reduce_mean(residual_square_lst) /\
            tf.reduce_mean(return_square_lst)
        return r2
