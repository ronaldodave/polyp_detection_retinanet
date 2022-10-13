import keras_retinanet
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras_retinanet import models, losses
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters, parse_pyramid_levels
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.callbacks import RedirectModel


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model
    
def create_models(backbone_retinanet, num_classes, weights,
                  freeze_backbone=False, lr=1e-5, config=None):
    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    pyramid_levels = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()
    if config and 'pyramid_levels' in config:
        pyramid_levels = parse_pyramid_levels(config)
    
    training_model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier, pyramid_levels=pyramid_levels), weights=weights, skip_mismatch=True)

    # make prediction model
    prediction_model = retinanet_bbox(model=training_model, anchor_params=anchor_params, pyramid_levels=pyramid_levels)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.SGD(lr=lr, decay=1e-4, momentum=0.9),
      
    )
    return training_model, prediction_model

def create_callbacks(training_model, prediction_model, validation_generator, model_path, log_file):
    callbacks = []
    tensorboard_callback = None
    evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback, weighted_average=True)
    evaluation = RedirectModel(evaluation, prediction_model)
    callbacks.append(evaluation)
    checkpoint = keras.callbacks.ModelCheckpoint(model_path,
                                                verbose=1,
                                                save_best_only=True,
                                                monitor="mAP",
                                                mode='max'
                                                )
    checkpoint = RedirectModel(checkpoint, training_model)
    callbacks.append(checkpoint)
    callbacks.append(keras.callbacks.CSVLogger(log_file))
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'val_loss',
        factor     = 0.1,
        patience   = 5,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0 
    ))
    return callbacks

def plot_history(sejarah, fig, subtitle, epochs):
    epoch_list = list(range(1,epochs+1)) 
    regression_loss = sejarah['regression_loss']
    classification_loss = sejarah['classification_loss']
    val_regression_loss =sejarah['val_regression_loss']
    val_classification_loss = sejarah['val_classification_loss']
    loss = sejarah['loss']
    val_loss = sejarah['val_loss']
    val_mAP = sejarah['mAP']
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(13,20))
    f.subplots_adjust(hspace=0.3)
    t = f.suptitle(subtitle, fontsize=17)
    
    ax1.plot(epoch_list, classification_loss, label='Train Classification Loss')
    ax1.plot(epoch_list, val_classification_loss, label='Validation Classification Loss')
    ax1.set_xticks(np.arange(1, epochs+1, 1))
    ax1.set_ylim(0,1.3)
    ax1.set_ylabel('Focal Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Classification Loss')
    l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, regression_loss, label='Train Regression Loss')
    ax2.plot(epoch_list, val_regression_loss, label='Validation Regression Loss')
    ax2.set_xticks(np.arange(1, epochs+1, 1))
    ax2.set_ylim(0,3)
    ax2.set_ylabel('Smooth L1 Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Regression Loss')
    l2 = ax2.legend(loc="best")

    ax3.plot(epoch_list, loss, label='Train Loss')
    ax3.plot(epoch_list, val_loss, label='Validation Loss')
    ax3.set_xticks(np.arange(1, epochs+1, 1))
    ax3.set_ylim(0,4)
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_title('Total Loss (Classification Loss + Regression Loss)')
    l3 = ax3.legend(loc="best")
    
    ax4.plot(epoch_list, val_mAP, label='Validation AP')
    ax4.set_xticks(np.arange(1, epochs+1, 1))
    ax4.set_ylim(0,1)
    ax4.set_ylabel('AP')
    ax4.set_xlabel('Epoch')
    ax4.set_title('Average Precision (AP)')
    l4 = ax4.legend(loc="best")
    plt.savefig(fig)