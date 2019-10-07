#!/bin/python -f
import numpy as np;
import cv2
import matplotlib.pyplot as plt

train_accs = np.load('./p0_batch_test_model_01_train_accs.npy')
train_accs_batch = np.load('./p0_batch_test_model_01_train_accs_batch.npy')
val_accs = np.load('./p0_batch_test_model_01_val_accs.npy')
val_accs_batch = np.load('./p0_batch_test_model_01_val_accs_batch.npy')

fig0, axes0 = plt.subplots(1, 2, figsize=(16,4))

axes0[0].plot(range(0,len(train_accs_batch)),train_accs_batch, label='With batch')
axes0[0].plot(range(0,len(train_accs)),train_accs, label='Without batch')
axes0[0].set_xlabel('Epochs')
axes0[0].set_ylabel('Accuracy')
axes0[0].set_xticks(range(0,len(train_accs),1))
axes0[0].set_ylim([0.85,1.01])
axes0[0].set_title('Accuracy on Training Dataset')
axes0[0].legend(loc=4)
axes0[0].grid(True)
axes0[1].plot(range(0,len(val_accs_batch)),val_accs_batch, label='With batch')
axes0[1].plot(range(0,len(val_accs)),val_accs, label='Without batch')
axes0[1].set_xlabel('Epochs')
axes0[1].set_ylabel('Accuracy')
axes0[1].set_xticks(range(0,len(val_accs),1))
axes0[1].set_ylim([0.85,1])
axes0[1].set_title('Accuracy on Validation Dataset')
axes0[1].legend(loc=4)
axes0[1].grid(True)

plt.show()

conv2_betas = np.load('./p0_batch_test_model_01_conv2_betas.npy')
conv2_gammas = np.load('./p0_batch_test_model_01_conv2_gammas.npy')
conv2_pop_means = np.load('./p0_batch_test_model_01_conv2_pop_means.npy')
conv2_pop_vars = np.load('./p0_batch_test_model_01_conv2_pop_vars.npy')
conv2_batch_ms = np.load('./p0_batch_test_model_01_conv2_batch_ms.npy')
conv2_batch_vs = np.load('./p0_batch_test_model_01_conv2_batch_vs.npy')

fig1, axes1 = plt.subplots(1, 6, figsize=(20,4))
fig1.tight_layout()

#Select the feature map to display
j = 3

axes1[0].set_title("Batch Mean")
axes1[1].set_title("Batch Variance")
axes1[2].set_title("Population Mean")
axes1[3].set_title("Population Variance")
axes1[4].set_title("Beta")
axes1[5].set_title("Gamma")
axes1[0].plot(conv2_batch_ms[:,j])
axes1[1].plot(conv2_batch_vs[:,j])
axes1[2].plot(conv2_pop_means[:,j])
axes1[3].plot(conv2_pop_vars[:,j])
axes1[4].plot(conv2_betas[:,j])
axes1[5].plot(conv2_gammas[:,j])
for ax in axes1:
    ax.grid(True)

plt.show()
