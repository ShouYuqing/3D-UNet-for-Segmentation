from keras.layers import Input, BatchNormalization, MaxPool3D, Conv3D, UpSampling3D, Concatenate, Activation
from keras.models import Model
from keras.optimizers import Adam
from losses import *

def unet(label_num=1, pretrained_weights=None):

    inputs = Input((64, 64, 64, 1))

    '''downsample'''
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(inputs)
    batc1 = BatchNormalization(axis=-1)(conv1)
    acti1 = Activation('relu')(batc1)
    conv2 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(acti1)
    batc2 = BatchNormalization(axis=-1)(conv2)
    acti2 = Activation('relu')(batc2)
    maxp1 = MaxPool3D(2)(acti2)


    conv3 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(maxp1)
    batc3 = BatchNormalization(axis=-1)(conv3)
    acti3 = Activation('relu')(batc3)
    conv4 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(acti3)
    batc4 = BatchNormalization(axis=-1)(conv4)
    acti4 = Activation('relu')(batc4)
    maxp2 = MaxPool3D(2)(acti4)


    conv5 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(maxp2)
    batc5 = BatchNormalization(axis=-1)(conv5)
    acti5 = Activation('relu')(batc5)
    conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(acti5)
    batc6 = BatchNormalization(axis=-1)(conv6)
    acti6 = Activation('relu')(batc6)
    maxp3 = MaxPool3D(2)(acti6)

    conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(maxp3)
    batc7 = BatchNormalization(axis=-1)(conv7)
    acti7 = Activation('relu')(batc7)
    conv8 = Conv3D(128, 3, padding='same', kernel_initializer='he_normal')(acti7)
    batc8 = BatchNormalization(axis=-1)(conv8)
    acti8 = Activation('relu')(batc8)


    '''upsample'''
    upsa1 = UpSampling3D(2)(acti8)
    # print('upsam1 shape: ', upsam1.shape)
    merg1 = Concatenate(axis=-1)([conv6, upsa1])
    conv9 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merg1)
    batc9 = BatchNormalization(axis=-1)(conv9)
    acti9 = Activation('relu')(batc9)
    conv10 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(acti9)
    batc10 = BatchNormalization(axis=-1)(conv10)
    acti10 = Activation('relu')(batc10)

    upsa2 = UpSampling3D(2)(acti10)
    merg2 = Concatenate(axis=-1)([conv4, upsa2])
    conv11 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merg2)
    batc11 = BatchNormalization(axis=-1)(conv11)
    acti11 = Activation('relu')(batc11)
    conv12 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(acti11)
    batc12 = BatchNormalization(axis=-1)(conv12)
    acti12 = Activation('relu')(batc12)

    upsa3 = UpSampling3D(2)(acti12)
    merg3 = Concatenate(axis=-1)([conv2, upsa3])
    conv13 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(merg3)
    batc13 = BatchNormalization(axis=-1)(conv13)
    acti13 = Activation('relu')(batc13)
    conv14 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(acti13)
    convol = Conv3D(label_num, 1, activation='softmax')(conv14)


    model = Model(inputs=inputs, outputs=convol)
    #model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-4), loss=diceLoss, metrics=[dice_coef])
    #model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model