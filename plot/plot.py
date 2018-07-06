import csv
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = 'data/'
OUT_DIR = 'img/'


def _extract_data(file_path):
    with open(file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        steps = []
        values = []
        for row in reader:
            steps.append(float(row['Step']))
            values.append(float(row['Value']))
    return steps, values


def _smooth_i(x,window_len=11,window='hanning'):
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def _smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

'''
Plot different models loss on the same graph
'''


def plot_loss(in_files, out_file, names, colors, smooth_val, limits, dims=(640, 480), yticks=None, xticks=None, styles=None):
    num_models = len(in_files)
    limits_x = []
    limits_y = []
    for i in range(num_models):
        steps, values = _extract_data(in_files[i])
        limits_x.append(max(steps))
        limits_y.append(max(values))

        if smooth_val[i] != -1:
            values = _smooth_i(values, smooth_val[i])

        if styles != None:
            plt.plot(steps, values[:1000], color=colors[i], alpha=1.0, label=names[i], linewidth=styles[i])
        else:
            plt.plot(steps, values[:1000], color=colors[i], alpha=1.0, label=names[i])

    # MIN
    if limits[2] == -1:
        plt.xlim(xmin=0)
    else:
        plt.xlim(xmin=limits[2])

    if limits[3] == -1:
        plt.ylim(ymin=0)
    else:
        plt.ylim(ymin=limits[3])

    #MAX
    if limits[0] == -1:
        plt.xlim(xmax=max(limits_x))
    else:
        plt.xlim(xmax=limits[0])
    if limits[1] == -1:
        plt.ylim(ymax=max(limits_y))
    else:
        plt.ylim(ymax=limits[1])

    plt.grid(linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if len(names) > 1:
        plt.legend(names)

    if yticks is not None:
        plt.yticks(yticks)
    if xticks is not None:
        plt.xticks(xticks)

    # set dims
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(dims[0] / float(DPI), dims[1] / float(DPI))

    plt.savefig(out_file, bbox_inches='tight')
    plt.gcf().clear()


'''
Plot different models acc on the same graph
'''


def plot_acc(in_files, out_file, names, colors, smooth_val, limits, dims=(640, 480), yticks=None):
    num_models = len(in_files)
    limits_x = []
    limits_y = []
    for i in range(num_models):
        steps, values = _extract_data(in_files[i])
        limits_x.append(max(steps))
        limits_y.append(max(values))

        if smooth_val[i] != -1:
            values = _smooth_i(values, smooth_val[i])
        plt.plot(steps, values[:1000], color=colors[i], alpha=1.0, label=names[i])

    # MIN
    if limits[2] == -1:
        plt.xlim(xmin=0)
    else:
        plt.xlim(xmin=limits[2])

    if limits[3] == -1:
        plt.ylim(ymin=0)
    else:
        plt.ylim(ymin=limits[3])

    # MAX
    if limits[0] == -1:
        plt.xlim(xmax=max(limits_x))
    else:
        plt.xlim(xmax=limits[0])

    if limits[1] == -1:
        plt.ylim(ymax=max(limits_y))
    else:
        plt.ylim(ymax=limits[1])

    plt.grid(linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if len(names) > 1:
        plt.legend(names)

    if yticks is not None:
        plt.yticks(yticks)

    # set dims
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(dims[0] / float(DPI), dims[1] / float(DPI))

    plt.savefig(out_file, bbox_inches='tight')
    plt.gcf().clear()

# Best model Train + Test
smooth_var = 10
# BLUE, RED
colors = [(57/255,106/255,177/255), (204/255,37/255,41/255)]
plot_loss([DATA_DIR + "resnet18_train_loss.csv", DATA_DIR + "resnet18_val_loss.csv"],
          OUT_DIR + 'resnet18_all_loss.png',
          ["Training", "Validation"],
          colors,
          [-1, -1],
          [-1, -1, 0, 0.1],
          (640, 400))

ticks = np.arange(0.8, 0.98, 0.04)
plot_acc([DATA_DIR + "resnet18_train_acc.csv", DATA_DIR + "resnet18_val_acc.csv"],
         OUT_DIR + 'resnet18_all_acc.png',
         ["Training", "Validation"],
         colors,
         [-1, -1],
         [-1, 0.96, 0, 0.8],
         (640, 400),
         ticks)



# SEGNET
# Best model Train + Test
smooth_var = 8
# BLUE, RED
colors = ['dodgerblue','darkblue']
plot_loss([DATA_DIR + "segnet_train_loss.csv", DATA_DIR + "segnet_val_loss.csv"],
          OUT_DIR + 'segnet_all_loss.png',
          ["Training", "Validation"],
          colors,
          [-1, smooth_var],
          [-1, 0.6, -1, -1],
          (640, 400),
          styles=[1,1])


smooth_var =15

# REDNET
smooth_var = 8
colors = ['limegreen', 'darkgreen']
ticks = np.arange(0, 4500, 500)
plot_loss([DATA_DIR + "rednet50_train_loss.csv", DATA_DIR + "rednet50_val_loss.csv"],
          OUT_DIR + 'rednet50_all_loss.png',
          ["Training", "Validation"],
          colors,
          [-1, smooth_var],
          [-1, 0.6, -1, -1],
          (640, 400),
          styles=[1,1])

# SEGNET + REDNET
smooth_var = 8
colors = [(57/255,106/255,177/255),(218/255,124/255,48/255), (204/255,37/255,41/255), (62/255,150/255,81/255)]

plot_loss([DATA_DIR + "segnet_train_loss.csv", DATA_DIR + "rednet50_train_loss.csv", DATA_DIR + "segnet_val_loss.csv", DATA_DIR + "rednet50_val_loss.csv"],
          OUT_DIR + 'fcn_all_loss.png',
          ["SegNet (Training)",  "RedNet50 (Training)", "SegNet (Validation)", "RedNet50 (Validation)",],
          colors,
          [-1, -1, smooth_var, smooth_var],
          [3000, 0.6, -1, -1],
          (640, 400))
