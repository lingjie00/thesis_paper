"""Implement Conditional Network."""
import tensorflow as tf
import tensorflow.keras as keras

"""
Model structure:
=macro data= -> |RNN| + =firm data= -> |FFN|

Input data:
    - macro data (time series)
    - firm data (time series, entity level)

Output:
    - latent variables
"""


class ConditionalModel(keras.Model):
    """Builds a Conditional model.

    Procedure:
        Pre-process data
        -> Macro data RNN layer
        -> (Macro + firm data) Dense layer
        -> compute moments
    """

    def __init__(self,
                 LSTM_units: int = 128,
                 Dropout_rate: float = 0.):
        """Init model."""
        super().__init__(name="Condtional")
        self.lstm = keras.layers.LSTM(units=LSTM_units, name="Moment RNN")
        self.output_dense = keras.layers.Dense(units=8, activation=tf.nn.tanh,
                                               name="Conditional g")
        self.dropout = keras.layers.Dropout(Dropout_rate)

    def call(self, inputs: list):
        """Defines the network architecture.

        :param inputs: Input data = [macroeconomic data, firm data]
        macroeconomic data: Time * macro feature dimension * 1
        firm data: Time * No firms * firm feature dimension + 1
                    + 1 for the Return data
        :param mask: Mask years without valid return data
        :param normalize: moments

        """
        ###################
        # Data processing #
        ###################
        macro_data, firm_data = inputs
        firm_data = firm_data[:, :, 1:]

        ########################
        # Macro data RNN layer #
        ########################
        # Macro data -> Dropout -> LSTM
        h = self.dropout(macro_data)
        h = self.lstm(h)

        ###################################
        # (Macro + firm data) Dense layer #
        ###################################
        # Merge macro data with firm data
        # We need to repeat macro data for each firm
        # Therefore, the final macro data dimension:
        #   Time * No firms * macro feature
        num_firms = firm_data.shape[1]
        h = tf.expand_dims(h, axis=1)  # insert Tensor of 1s in axis
        h = tf.tile(h, [1, num_firms, 1])  # repeat macro data for each firm

        # The input data will be dim:
        #  None * (macro feature + firm feature)
        concat = tf.concat([firm_data, h], axis=2)

        ###########
        # moments #
        ###########
        # dimension: num condition moment * time * num firms
        # TODO: understand why moment is calculated as such
        g = self.output_dense(concat)
        g = tf.transpose(g, perm=[2, 0, 1])

        return g
