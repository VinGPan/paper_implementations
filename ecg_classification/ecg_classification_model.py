# Implementation of "Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks", 2017,
# Pranav Rajpurkar, Awni Y. Hannun, Masoumeh Haghpanahi, Codie Bourn, Andrew Y. Ng
#
# https://arxiv.org/pdf/1707.01836.pdf

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Input, Conv1D, BatchNormalization, Activation, Dropout, MaxPool1D, Add, Dense, Flatten
from keras.models import Model


class ECGClassificationModel:
    def __init__(self, input_size, output_classes):
        """
        :param input_size: This is epoch size of ECG data. For example, qt 200Hz data if we have label for every two
        seconds then this will be equal to 200 * 2 = 400
        :param output_classes: number of ECG classes.
        """

        assert input_size >= 200, "Input size should be at least 200 because there are 8 max pooling layers in this " \
                                  "network"
        self.input_size = input_size
        self.output_classes = output_classes

        self.kernel_size = 16       # Conv1D param as per the paper
        self.filters = 64           # Conv1D param as per the paper
        self.total_res_blocks = 16  # Total number of residual blocks as per the paper

        self.dropout_rate = 0.5     # We have assumed this to be 0.5 since this is not specified in the paper

    @staticmethod
    def pad_depth(x, desired_channels):
        # The paper does not specify how to handle the dimension mis-match while creating the residual block. Here, we
        # are using (1X1) Convolution filters to match-up the channels
        x = Conv1D(desired_channels, 1, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        return x

    def initial_block(self, input, k):
        x = Conv1D(filters=self.filters * k, kernel_size=self.kernel_size, padding='same',
                   kernel_initializer='he_normal')(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    # First residual block with 2 convolutional layers in it.
    # All convolutional layers all will have a filter length of 16 and have 64k filters
    def residual_block_type_1(self, input, k, res_id):
        x = Conv1D(filters=self.filters * k, kernel_size=self.kernel_size, padding='same',
                   kernel_initializer='he_normal')(input)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Conv1D(filters=self.filters * k, kernel_size=self.kernel_size, padding='same',
                   kernel_initializer='he_normal')(x)

        x = Add()([x, input])
        return x

    # Residual block with 2 convolutional layers in it. This should be used for blocks 2 to 16.
    # All convolutional layers all will have a filter length of 16 and have 64k filters
    def residual_block_type_2(self, input, k, subsample, pad_channels, res_id):
        short_cut = input

        # Subsample Input using max pooling. Subsampling is done every alternate block
        if subsample:
            input = MaxPool1D(2, 2)(input)
            # When a residual block subsamples the input, the corresponding shortcut connections also subsample
            # their input using a Max Pooling operation with the same subsample factor
            short_cut = input

        # Whenever k increases we need to pad the 'shortcut', else channel dimensions do not match
        if pad_channels:
            short_cut = self.pad_depth(input, self.filters * k)

        x = BatchNormalization()(input)
        x = Activation('relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Conv1D(filters=self.filters * k, kernel_size=self.kernel_size, padding='same',
                   kernel_initializer='he_normal')(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Conv1D(filters=self.filters * k, kernel_size=self.kernel_size, padding='same',
                   kernel_initializer='he_normal')(x)

        x = Add()([x, short_cut])
        return x

    def final_block(self, input):
        x = BatchNormalization()(input)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(self.output_classes)(x)
        x = Activation('softmax')(x)
        return x

    def get_model(self):
        input = Input(shape=(self.input_size, 1))

        x = self.initial_block(input, 1)

        # Add 16 residual blocks
        k = 1
        subsample = False
        pad_channels = False
        for res_id in range(1, self.total_res_blocks+1):
            if res_id == 1:
                x = self.residual_block_type_1(x, k, res_id)
            else:
                x = self.residual_block_type_2(x, k, subsample, pad_channels, res_id)

            # The convolutional layers 64k filters, where k starts out as 1 and is incremented every 4-th residual block
            if (res_id % 4) == 0:
                k += 1
                pad_channels = True
            else:
                pad_channels = False

            # Every alternate residual block subsamples its inputs
            subsample = res_id % 2 == 0

        y = self.final_block(x)

        model = Model(input, y)

        model.compile(optimizer='adam', loss='categorical_crossentropy')

        return model

if __name__ == '__main__':
    input_size = 200  # ie This is epoch size of ECG data. For example, qt 200Hz data if we have label for every two
    # seconds then this will be equal to 200 * 2 = 400
    output_classes = 14  # Number of ECG classes.
    model = ECGClassificationModel(input_size, output_classes)
    model = model.get_model()
    model.summary()
