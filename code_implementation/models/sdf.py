"""Implement Stochastic Discount Factor (SDF) Network."""
import tensorflow as tf
import tensorflow.keras as keras

"""
Model structure:
=macro data= -> |RNN| + =firm data= -> |FFN| -> |SDF|

Input data:
    - macro data (time series)
    - firm data (time series, entity level)

Output:
    - SDF
"""


class SDFModel(keras.Model):
    """Builds a SDF model.

    Procedure:
        Pre-process data
        -> Macro data RNN layer
        -> (Macro + firm data) Dense layer
        -> compute SDF
    """

    def __init__(self,
                 LSTM_units: int = 4,
                 Dense_units: int = 64,
                 Dropout_rate: float = 0.95):
        """Init model."""
        super().__init__(name="SDF")
        self.lstm = keras.layers.LSTM(units=LSTM_units, name="State_RNN")
        self.dense1 = keras.layers.Dense(units=Dense_units, activation="relu",
                                         name="SDF_dense1")
        self.dense2 = keras.layers.Dense(units=Dense_units, activation="relu",
                                         name="SDF_dense2")
        self.dense_output = keras.layers.Dense(units=1, activation="linear",
                                               name="SDF_w")
        self.dropout = keras.layers.Dropout(Dropout_rate)

    def call(self,
             inputs: list,
             normalize: bool = True):
        """Defines the network architecture.

        :param inputs: Input data = [macroeconomic data, firm data]
        macroeconomic data: Time * macro feature dimension * 1
        firm data: Time * No firms * firm feature dimension + 1
                    + 1 for the Return data
        :param normalize: Normalise SDF

        """
        ###################
        # Data processing #
        ###################
        macro_data, firm_data, return_data, mask = inputs

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

        # mask input data
        masked_concat = tf.boolean_mask(concat, mask=mask)

        # Training macro + firm data
        w = self.dense1(masked_concat)
        w = self.dropout(w)
        w = self.dense2(w)
        w = self.dropout(w)
        w = self.dense_output(w)

        ###############
        # Compute SDF #
        ###############
        # TODO: understand how to compute SDF
        # mask Return data
        masked_return_data = tf.boolean_mask(return_data, mask=mask)
        # weighted masked Return data with dense output
        weighted_return_data = masked_return_data * w

        mask = tf.cast(mask, "int32")
        Ti = tf.reduce_sum(mask, axis=1)  # length of Time
        weighted_return_data_split = tf.split(weighted_return_data,
                                              num_or_size_splits=Ti)

        # compute SDF
        sum_lst = []
        for item in weighted_return_data_split:
            item_sum = tf.reduce_sum(item, keepdims=True)
            sum_lst.append(item_sum)
        SDF = tf.concat(sum_lst, axis=0)

        # normalize SDF
        if normalize:
            mean_Ni = tf.reduce_mean(Ti)
            mean_Ni = tf.cast(mean_Ni, "float")
            Ti = tf.cast(Ti, "float")
            Ti = tf.expand_dims(Ti, axis=1)
            SDF = SDF / Ti % mean_Ni

        SDF += 1
        SDF = tf.where(tf.math.is_nan(SDF), 0, SDF)
        return SDF
