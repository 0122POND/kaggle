
# import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
import seaborn as sns
from cycler import cycler
from IPython.display import display
import datetime
import scipy.stats
import math
import random

from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Input, InputLayer, Add, Concatenate
from tensorflow.keras.utils import plot_model

plt.rcParams['axes.prop_cycle'] = cycler(color=['#ffd700'] +
                                         plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])



print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#TensorFlowがGPUを認識しているか確認
from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# Plot training history
def plot_history(history, *, n_epochs=None, plot_lr=False, title=None, bottom=None, top=None):
    """Plot (the last n_epochs epochs of) the training history
    
    Plots loss and optionally val_loss and lr."""
    plt.figure(figsize=(15, 6))
    from_epoch = 0 if n_epochs is None else max(len(history['loss']) - n_epochs, 0)
    
    # Plot training and validation losses
    plt.plot(np.arange(from_epoch, len(history['loss'])), history['loss'][from_epoch:], label='Training loss')
    try:
        plt.plot(np.arange(from_epoch, len(history['loss'])), history['val_loss'][from_epoch:], label='Validation loss')
        best_epoch = np.argmin(np.array(history['val_loss']))
        best_val_loss = history['val_loss'][best_epoch]
        if best_epoch >= from_epoch:
            plt.scatter([best_epoch], [best_val_loss], c='r', label=f'Best val_loss = {best_val_loss:.5f}')
        if best_epoch > 0:
            almost_epoch = np.argmin(np.array(history['val_loss'])[:best_epoch])
            almost_val_loss = history['val_loss'][almost_epoch]
            if almost_epoch >= from_epoch:
                plt.scatter([almost_epoch], [almost_val_loss], c='orange', label='Second best val_loss')
    except KeyError:
        pass
    if bottom is not None: plt.ylim(bottom=bottom)
    if top is not None: plt.ylim(top=top)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    if title is not None: plt.title(title)
        
    # Plot learning rate
    if plot_lr and 'lr' in history:
        ax2 = plt.gca().twinx()
        ax2.plot(np.arange(from_epoch, len(history['lr'])), np.array(history['lr'][from_epoch:]), color='g', label='Learning rate')
        ax2.set_ylabel('Learning rate')
        ax2.legend(loc='upper right')
        
    plt.show()

# train data and test data
train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv') #データをダウンロードしたディレクトリの場所を指定
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv') #データをダウンロードしたディレクトリの場所を指定


# top three feature interactions(上位3位の特徴量相互作用)
plt.rcParams['axes.facecolor'] = 'k'
plt.figure(figsize=(11, 5))
cmap = ListedColormap(["#ffd700", "#0057b8"])

ax = plt.subplot(1, 3, 1)
ax.scatter(train['f_02'], train['f_21'], s=1,
           c=train.target, cmap=cmap)
ax.set_xlabel('f_02')
ax.set_ylabel('f_21')
ax.set_aspect('equal')
ax0 = ax

ax = plt.subplot(1, 3, 2, sharex=ax0, sharey=ax0)
ax.scatter(train['f_05'], train['f_22'], s=1,
           c=train.target, cmap=cmap)
ax.set_xlabel('f_05')
ax.set_ylabel('f_22')
ax.set_aspect('equal')

ax = plt.subplot(1, 3, 3, sharex=ax0, sharey=ax0)
ax.scatter(train['f_00'] + train['f_01'], train['f_26'], s=1,
           c=train.target, cmap=cmap)
ax.set_xlabel('f_00 + f_01')
ax.set_ylabel('f_26')
ax.set_aspect('equal')

plt.tight_layout(w_pad=1.0)
plt.show()
plt.rcParams['axes.facecolor'] = '#0057b8' # blue



#feature engineering(特徴量エンジニアリング)
features = [f for f in test.columns if f != 'id' and f != 'f_27']
float_features = [f for f in features if test[f].dtype == float]

for df in [train, test]:
    # Extract the 10 letters of f_27 into individual features
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
        
    # unique_characters feature is from https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))
    
    # Feature interactions: create three ternary features
    # Every ternary feature can have the values -1, 0 and +1
    df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
    df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
    
features = [f for f in test.columns if f != 'id' and f != 'f_27']
float_features = [f for f in features if test[f].dtype == float]
int_features = [f for f in features if test[f].dtype == int and f.startswith('f')]
ch_features = [f for f in features if f.startswith('ch')]
test[features].head(2)

# The model
def my_model():
    
    operation_lebel_seed=42
    initializer=tf.keras.initializers.GlorotUniform(seed=operation_lebel_seed)
    activation = 'silu'
    inputs = Input(shape=(len(features)))
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(40e-6),
              activation=activation,kernel_initializer=initializer
             )(inputs)
    x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(40e-6),
              activation='PReLU',kernel_initializer=initializer
             )(x)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(40e-6),
              activation='ReLU',kernel_initializer=initializer
             )(x)
    x = Dense(32, kernel_regularizer=tf.keras.regularizers.l2(40e-6),
              activation=activation,kernel_initializer=initializer
             )(x)
    x = Dense(16, kernel_regularizer=tf.keras.regularizers.l2(40e-6),
              activation=activation,kernel_initializer=initializer
             )(x)
    x = Dense(1, #kernel_regularizer=tf.keras.regularizers.l2(1e-6),
              activation='sigmoid',
             )(x)
    model = Model(inputs, x)
    return model

plot_model(my_model(), show_layer_names=False, show_shapes=True)



# Cross validation
#%%time
# Cross-validation of the classifier

EPOCHS = 200
EPOCHS_COSINEDECAY = 150
CYCLES = 1
VERBOSE = 0 # set to 0 for less output, or to 2 for more output
DIAGRAMS = True
USE_PLATEAU = False
BATCH_SIZE = 2048
ONLY_FIRST_FOLD = False
global_seed = 42
# see https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
np.random.seed(global_seed)
random.seed(global_seed)
tf.random.set_seed(global_seed)

def fit_model(X_tr, y_tr, X_va=None, y_va=None, run=0):
    """Scale the data, fit a model, plot the training history and optionally validate the model
    
    Returns a trained instance of tensorflow.keras.models.Model.
    
    As a side effect, updates y_va_pred, history_list and score_list.
    """
    global y_va_pred
    start_time = datetime.datetime.now()
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    
    if X_va is not None:
        X_va = scaler.transform(X_va)
        validation_data = (X_va, y_va)
    else:
        validation_data = None

    # Define the learning rate schedule and EarlyStopping
    lr_start=0.01

    if USE_PLATEAU and X_va is not None: # use early stopping
        epochs = EPOCHS
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, 
                               patience=4, verbose=VERBOSE)
        es = EarlyStopping(monitor="val_loss",
                           patience=12, 
                           verbose=1,
                           mode="min", 
                           restore_best_weights=True)
        callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]

    else: # use cosine learning rate decay rather than early stopping
        epochs = EPOCHS_COSINEDECAY
        lr_end = 0.00005
        def cosine_decay(epoch):
            # w decays from 1 to 0 in every cycle
            # epoch == 0                  -> w = 1 (first epoch of cycle)
            # epoch == epochs_per_cycle-1 -> w = 0 (last epoch of cycle)
            epochs_per_cycle = epochs // CYCLES
            epoch_in_cycle = epoch % epochs_per_cycle
            if epochs_per_cycle > 1:
                w = (1 + math.cos(epoch_in_cycle / (epochs_per_cycle-1) * math.pi)) / 2
            else:
                w = 1
            return w * lr_start + (1 - w) * lr_end

        lr = LearningRateScheduler(cosine_decay, verbose=0)
        callbacks = [lr, tf.keras.callbacks.TerminateOnNaN()]
        
    # Construct and compile the model
    model = my_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_start),
                  metrics='AUC',
                  loss=tf.keras.losses.BinaryCrossentropy())

    # Train the model
    history = model.fit(X_tr, y_tr, 
                        validation_data=validation_data, 
                        epochs=epochs,
                        verbose=VERBOSE,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=callbacks)

    history_list.append(history.history)
    callbacks, es, lr, history = None, None, None, None
    
    if X_va is None:
        print(f"Training loss: {history_list[-1]['loss'][-1]:.4f}")
    else:
        lastloss = f"Training loss: {history_list[-1]['loss'][-1]:.4f} | Val loss: {history_list[-1]['val_loss'][-1]:.4f}"
        
        # Inference for validation
        y_va_pred = model.predict(X_va, batch_size=len(X_va), verbose=0)
        #oof_list[run][val_idx] = y_va_pred
        
        # Evaluation: Execution time, loss and AUC
        score = roc_auc_score(y_va, y_va_pred)
        print(f"Fold {run}.{fold} | {str(datetime.datetime.now() - start_time)[-12:-7]}"
              f" | {lastloss} | AUC: {score:.5f}")
        score_list.append(score)
        
        if DIAGRAMS and fold == 0 and run == 0:
            # Plot training history
            plot_history(history_list[-1], 
                         title=f"Learning curve (validation AUC = {score:.5f})",
                         plot_lr=True)

            # Plot y_true vs. y_pred
            plt.figure(figsize=(10, 4))
            plt.hist(y_va_pred[y_va == 0], bins=np.linspace(0, 1, 21),
                     alpha=0.5, density=True)
            plt.hist(y_va_pred[y_va == 1], bins=np.linspace(0, 1, 21),
                     alpha=0.5, density=True)
            plt.xlabel('y_pred')
            plt.ylabel('density')
            plt.title('OOF Predictions')
            plt.show()

    return model, scaler


print(f"{len(features)} features")
history_list = []
score_list = []
kf = KFold(n_splits=5)

for fold, (idx_tr, idx_va) in tqdm(enumerate(kf.split(train))):
    X_tr = train.iloc[idx_tr][features]
    X_va = train.iloc[idx_va][features]
    y_tr = train.iloc[idx_tr].target
    y_va = train.iloc[idx_va].target
    
    fit_model(X_tr, y_tr, X_va, y_va)
    if ONLY_FIRST_FOLD: break # we only need the first fold

print(f"OOF AUC:                       {np.mean(score_list):.5f}")


# submission
#%%time
# Create submission
print(f"{len(features)} features")

X_tr = train[features]
y_tr = train.target

pred_list = []
seeds_list = [0, 5, 10, 15, 20, 25, 30, 42, 50, 52]
for seed in seeds_list:
    # see https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    model, scaler = fit_model(X_tr, y_tr, run=seed)
    pred_list.append(scipy.stats.rankdata(model.predict(scaler.transform(test[features]),
                                                        batch_size=len(test), verbose=0)))
    print(f"{seed:2}", pred_list[-1])
print()
submission = test[['id']].copy()
submission['target'] = np.array(pred_list).mean(axis=0)
submission.to_csv('submission.csv', index=False)
submission