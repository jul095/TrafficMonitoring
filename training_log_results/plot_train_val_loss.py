#  ****************************************************************************
#  @plot_train_val_loss.py
#
#  This script is for generating the plots based on
#  the metric files for better visualization
#
#
#  @copyright (c) 2021 Elektronische Fahrwerksysteme GmbH. All rights reserved.
#  Dr.-Ludwig-Kraus-Straße 6, 85080 Gaimersheim, DE, https://www.efs-auto.com
#  ****************************************************************************

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser("Prepare Diagrams")

parser.add_argument('--metric_file', type=str, help='Metric file')

args = parser.parse_args()


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


experiment_metrics = load_json_arr(args.metric_file)

# print([[x['iteration'], if 'total_loss' in x] for x in experiment_metrics])


batch_size = 2
# batch_size = 4
# number_images = 12015
# number_images = 14015
number_images = 9802

fig, ax1 = plt.subplots()

ax1.set_xlabel('iterations')
ax1.set_ylabel('loss')
ax1.plot([x['iteration'] for x in experiment_metrics if 'total_loss' in x],
         [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], label="training loss")

ax1.plot([x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
         [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x], label="validation loss")

ax2 = ax1.twinx()
ax3 = ax1.twiny()

epochs = [(x['iteration'] * batch_size) / number_images for x in experiment_metrics if 'total_loss' in x]
epochs = [round(epoch, 0) for epoch in epochs]
epochs = list(dict.fromkeys(epochs))

ax3.set_xlabel('epochs')
ax3.set_xticks(epochs)

ax1.grid()

ax2.set_ylabel('learning rate')
ax2.plot([x['iteration'] for x in experiment_metrics if 'total_loss' in x],
         [x['lr'] for x in experiment_metrics if 'lr' in x], label="learning rate", color="green")
fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
# fig.legend(loc="upper right", ncol=1)
# ax2.legend(loc="upper right")
# ax1.legend(loc="upper left")

# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.tight_layout()
# plt.plot(
#    [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
#    [x['lr'] for x in experiment_metrics if 'lr' in x])

# plt.legend(['total_loss', 'validation_loss'], loc='upper right')
# plt.show()
fig.savefig(f"plot_loss.png", transparent=True, bbox_inches='tight', pad_inches=0)

fig2, ax1 = plt.subplots()
ax1.grid()
ax1.plot([x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
         [x['segm/AP'] for x in experiment_metrics if 'validation_loss' in x], label="AP", color="black")
ax1.plot([x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
         [x['segm/AP-transporter'] for x in experiment_metrics if 'validation_loss' in x], label="AP transporter",
         color="magenta")

ax1.plot([x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
         [x['segm/AP-person'] for x in experiment_metrics if 'validation_loss' in x], label="AP person", color="navy")

ax1.plot([x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
         [x['segm/AP-car'] for x in experiment_metrics if 'validation_loss' in x], label="AP car", color="goldenrod")
ax1.plot([x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
         [x['segm/AP-bicycle'] for x in experiment_metrics if 'validation_loss' in x], label="AP bicycle",
         color="darkcyan")

ax1.legend(loc="center right")
ax1.set_yticks(np.arange(0, 100, 10))
ax1.set_ylabel('Average Precision (AP)')

ax1.set_xlabel('iterations')

ax2 = ax1.twiny()
ax2.set_xlabel('epochs')

ax2.set_xticks(epochs[::2])

fig2.savefig("plot_ap.png", transparent=True, bbox_inches='tight', pad_inches=0)

# plt.show()
