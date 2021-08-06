"""Model training for GAN model."""
import tensorflow as tf
import tensorflow.keras as keras


class Trainer(object):
    """Trains the model.

    Training procedure:
        1. Train unconditional SDF loss
        2. Train moment condition
        3. Train conditional SDF loss
    """

    def __init__(self, loss: keras.losses.Loss):
        """Init trainer."""
        self.Loss = loss

    def train_sdf(self,
                  optimizer: keras.optimizers.Optimizer,
                  model: keras.models.Model,
                  inputs: list,
                  moment: tf.Tensor = None,
                  train_conditional: bool = False
                  ):
        """Take in model and data and conduct one training training."""
        firm_data, return_data, mask = inputs[1], inputs[2], inputs[3]
        if not train_conditional:
            moment = tf.ones(
                shape=(1, firm_data.shape[0], firm_data.shape[1])
            )
        with tf.GradientTape() as tape:
            sdf = model(inputs)
            loss = self.Loss(SDF=sdf,
                             moment=moment,
                             return_data=return_data,
                             mask=mask)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(
            gradients, model.trainable_variables))

        return loss, gradients

    def train_moment(self,
                     optimizer: keras.optimizers.Optimizer,
                     model: keras.models.Model,
                     inputs: list,
                     sdf: tf.Tensor
                     ):
        """Conduct a single training for the moment."""
        return_data, mask = inputs[2], inputs[3]
        with tf.GradientTape() as tape:
            moment = model(inputs)
            loss = self.Loss(SDF=sdf,
                             moment=moment,
                             return_data=return_data,
                             mask=mask)
            # we want to maximise the loss with conditional moments
            loss = -loss
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(
            gradients, model.trainable_variables))

        return loss, gradients

    def train(self,
              sdf_model: keras.models.Model,
              conditional_model: keras.models.Model,
              optimizer: keras.optimizers.Optimizer,
              inputs: list,
              sdf_epoches_unc: int,
              moment_epoches: int,
              sdf_epoches_cond: int,
              steps: int = 4,
              normalize: bool = False
              ):
        """Conduct training for the GAN model."""
        macro_data, firm_data, return_data, mask = inputs
        total_data = firm_data.shape[1]  # get the total number of rows

        print("==============Start training===========""")
        # 1. Train unconditional SDF
        for jj in range(sdf_epoches_unc):
            # one epoch
            losses = []
            sharpes = []
            for ii in range(0, total_data, steps):
                inputs = [
                    macro_data,
                    firm_data[:, ii:ii+steps, :],
                    return_data[:, ii:ii+steps],
                    mask[:, ii:ii+steps]
                ]
                loss, gradient = self.train_sdf(
                    optimizer=optimizer,
                    model=sdf_model,
                    inputs=inputs
                )
                if normalize:
                    sharpe_loss = self.Loss.sharpe_loss(
                        sdf_weight=sdf_model.trainable_weights[-2],
                        return_data=inputs[2],
                        mask=inputs[3],
                        normalize=normalize
                    )
                else:
                    sharpe_loss = self.Loss.sharpe_loss(
                        sdf=sdf_model(inputs)
                    )
                losses.append(loss)
                sharpes.append(sharpe_loss)
            print(f"""
                  unconditional SDF epochs {jj}
                  avg loss = {tf.reduce_mean(losses)}
                  avg Sharpe = {tf.reduce_mean(sharpes)}
                  """)

        # 2. Train moments
        for jj in range(moment_epoches):
            losses = []
            sharpes = []
            for ii in range(0, total_data, steps):
                inputs = [
                    macro_data,
                    firm_data[:, ii:ii+steps, :],
                    return_data[:, ii:ii+steps],
                    mask[:, ii:ii+steps]
                ]
                loss, gradient = self.train_moment(
                    optimizer=optimizer,
                    model=conditional_model,
                    inputs=inputs,
                    sdf=sdf_model(inputs)
                )
                if normalize:
                    sharpe_loss = self.Loss.sharpe_loss(
                        sdf_weight=sdf_model.trainable_weights[-2],
                        return_data=inputs[2],
                        mask=inputs[3],
                        normalize=normalize
                    )
                else:
                    sharpe_loss = self.Loss.sharpe_loss(
                        sdf=sdf_model(inputs)
                    )
                losses.append(loss)
                sharpes.append(sharpe_loss)
            print(f"""
                  moments epochs {jj}
                  avg loss = {tf.reduce_mean(losses)}
                  avg Sharpe = {tf.reduce_mean(sharpes)}
                  """)

        # 3. Train conditional SDF
        for jj in range(sdf_epoches_cond):
            losses = []
            sharpes = []
            # one epoch
            for ii in range(0, total_data, steps):
                inputs = [
                    macro_data,
                    firm_data[:, ii:ii+steps, :],
                    return_data[:, ii:ii+steps],
                    mask[:, ii:ii+steps]
                ]
                loss, gradient = self.train_sdf(
                    optimizer=optimizer,
                    model=sdf_model,
                    inputs=inputs,
                    moment=conditional_model(inputs)
                )
                if normalize:
                    sharpe_loss = self.Loss.sharpe_loss(
                        sdf_weight=sdf_model.trainable_weights[-2],
                        return_data=inputs[2],
                        mask=inputs[3],
                        normalize=normalize
                    )
                else:
                    sharpe_loss = self.Loss.sharpe_loss(
                        sdf=sdf_model(inputs)
                    )
                losses.append(loss)
                sharpes.append(sharpe_loss)
            print(f"""
                  conditional SDF epochs {jj}
                  avg loss = {tf.reduce_mean(losses)}
                  avg Sharpe = {tf.reduce_mean(sharpes)}
                  """)

        return loss

    def modified_train(self,
                       sdf_model: keras.models.Model,
                       conditional_model: keras.models.Model,
                       optimizer: keras.optimizers.Optimizer,
                       inputs: list,
                       sdf_epoches_unc: int,
                       epoches: int,
                       steps: int = 4,
                       normalize: bool = False
                       ):
        """Conduct training for the GAN model."""
        macro_data, firm_data, return_data, mask = inputs
        total_data = firm_data.shape[1]  # get the total number of rows

        print("==============Start unconditional training===========""")
        # 1. Train unconditional SDF
        for jj in range(sdf_epoches_unc):
            # one epoch
            losses = []
            sharpes = []
            for ii in range(0, total_data, steps):
                inputs = [
                    macro_data,
                    firm_data[:, ii:ii+steps, :],
                    return_data[:, ii:ii+steps],
                    mask[:, ii:ii+steps]
                ]
                loss, gradient = self.train_sdf(
                    optimizer=optimizer,
                    model=sdf_model,
                    inputs=inputs
                )
                if normalize:
                    sharpe_loss = self.Loss.sharpe_loss(
                        sdf_weight=sdf_model.trainable_weights[-2],
                        return_data=inputs[2],
                        mask=inputs[3],
                        normalize=normalize
                    )
                else:
                    sharpe_loss = self.Loss.sharpe_loss(
                        sdf=sdf_model(inputs)
                    )
                losses.append(loss)
                sharpes.append(sharpe_loss)
            print(f"""
                  unconditional SDF epochs {jj}
                  avg loss = {tf.reduce_mean(losses)}
                  avg Sharpe = {tf.reduce_mean(sharpes)}
                  """)

        print("==============Start training===========""")
        for jj in range(epoches):
            # one epoch
            sdf_losses, moment_losses = [], []
            sharpes = []
            for ii in range(0, total_data, steps):
                inputs = [
                    macro_data,
                    firm_data[:, ii:ii+steps, :],
                    return_data[:, ii:ii+steps],
                    mask[:, ii:ii+steps]
                ]
                # Train moments
                moment_loss, gradient = self.train_moment(
                    optimizer=optimizer,
                    model=conditional_model,
                    inputs=inputs,
                    sdf=sdf_model(inputs)
                )
                # Train SDF
                sdf_loss, gradient = self.train_sdf(
                    optimizer=optimizer,
                    model=sdf_model,
                    inputs=inputs,
                    moment=conditional_model(inputs)
                )
                # generate Sharpe loss
                if normalize:
                    sharpe_loss = self.Loss.sharpe_loss(
                        sdf_weight=sdf_model.trainable_weights[-2],
                        return_data=inputs[2],
                        mask=inputs[3],
                        normalize=normalize
                    )
                else:
                    sharpe_loss = self.Loss.sharpe_loss(
                        sdf=sdf_model(inputs)
                    )
                sdf_losses.append(sdf_loss)
                moment_losses.append(moment_loss)
                sharpes.append(sharpe_loss)
            print(f"""
                  epochs {jj}
                  avg SDF loss = {tf.reduce_mean(sdf_losses)}
                  avg moment loss = {tf.reduce_mean(moment_losses)}
                  avg Sharpe = {tf.reduce_mean(sharpes)}
                  """)
