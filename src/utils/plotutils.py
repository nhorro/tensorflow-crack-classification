import matplotlib.pyplot as plt
import numpy
import pandas as pd

def plot_learning_curves_from_history_file(filename):
    history = pd.read_csv(filename)
    hv = history.values
    epoch=hv[:,0]
    acc=hv[:,1]
    loss=hv[:,2]
    val_acc=hv[:,3]
    val_loss=hv[:,4]
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(epoch,acc,epoch,val_acc)
    axes[0].set_title('model accuracy')
    axes[0].grid(which="Both")
    axes[0].set_ylabel('accuracy')
    axes[0].set_xlabel('epoch')
    axes[0].legend(['train', 'test'], loc='lower right')
    axes[1].plot(epoch,loss,epoch,val_loss)
    axes[1].set_title('model loss')
    axes[1].grid(which="Both")
    axes[1].set_ylabel('loss')
    axes[1].set_xlabel('epoch')
    axes[1].legend(['train', 'test'], loc='upper center')
    return fig