import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("error")    # For 'try' blocks

cnt = 1     # Index counter of the image file


def createPlot(data_set, coordinate_type = 'abscissa_map', plot_type = 'line_scatter_plot',
              title = '', x_label = '', y_label = ''):
    """
    Main function to plot values according to defined arguments
    :param plot_type: Used to define the type one intends to plot
    :type plot_type: str
    :param coordinate_type:  Used to define the coordinate type of the given data_set argument
    :type coordinate_type: str
    :param data_set: Samples for plotting
    :type data_set: list ((x1, y1), (x2, y2), ..., (xn, yn)), such that coordinate_type=='xy_pair',
        or dict {x1: (y1, y2, ..., yn), ..., xn: (y1, y2, ..., yn)}, such that coordinate_type=='abscissa_map'
    :type title: str
    :param title: Displayed plot title
    :param x_label: Abscissa name
    :type x_label: str
    :param y_label: Ordinate name
    """
    if not checkArgs(plot_type, coordinate_type):
        return

    if plot_type != 'bar_chart':
        plotGraph(plot_type, coordinate_type, data_set, title, x_label, y_label)
    else:
        barChart(data_set, title, y_label)


def checkArgs(plot, coord):
    if plot not in ('line_plot', 'scatter_plot', 'line_scatter_plot', 'bar_chart'):
        print("Invalid plot type\n")
        return False
    if coord not in ('xy_pair', 'abscissa_map'):
        print("Invalid coordinate type\n")
        return False
    if plot == 'bar_chart' and coord != 'abscissa_map':
        print("Invalid coordinate type: bar chart implies a dictionary\n")
        return False
    return True


def plotGraph(plot_type, coordinate_type, data_set, title, x_label, y_label):
    x, y = getSampleValues(coordinate_type, data_set)
    r = linregress(x, y)

    if plot_type == 'scatter_plot' or plot_type == 'line_scatter_plot':
        plt.plot(x, y, 'o')
        plt.xticks(x)
    if plot_type == 'line_plot' or plot_type == 'line_scatter_plot':
        plt.plot(x, r.intercept + r.slope * x, '-')

    plt.xlabel(x_label, color='#179c43')
    plt.ylabel(y_label, color='#179c43')
    plt.title(title, loc='center', fontdict={'fontsize': 16})
    fig = plt.gcf()

    try:
        plt.show()

    except UserWarning:
        print("UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, "
              "thus the figure cannot be displayed\n")
        global cnt
        fig.savefig('./images/fig' + str(cnt) + '.png')
        cnt += 1


def getSampleValues(coordinate_type, data_set):
    x_set, y_set = [], []

    if coordinate_type == 'xy_pair':
        for x, y in data_set:
            x_set += [x]
            y_set += [y]

    elif coordinate_type == 'abscissa_map':
        for x, v in data_set.items():
            for y in v:
                x_set += [x]
                y_set += [y]

    return np.array(x_set), np.array(y_set)


def barChart(data_set, title, y_label):
    labels = data_set.keys()

    size = len(data_set[next(iter(data_set))])
    width = 1/(size+1)          # width of the bars
    x = np.arange(len(labels))  # label locations

    fig, ax = plt.subplots()
    data = np.transpose([np.array(v) for v in data_set.values()])   # Cluster of classes
    bars = []

    for i in range(len(data)):
        bars += [ax.bar(x + (i-size/2+0.5)*width, data[i], width, label = 'type '+str(i+1))]

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for b in bars:
        for rectangle in b:
            height = rectangle.get_height()
            ax.annotate('{}'.format(height), xy=(rectangle.get_x() + rectangle.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    fig.tight_layout()
    fig = plt.gcf()

    try:
        plt.show()
    except UserWarning:
        print("UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, "
              "thus the figure cannot be displayed\n")
        global cnt
        fig.savefig('./images/fig' + str(cnt) + '.png')
        cnt += 1


#s1 = ((1, 2), (2, 4), (3, 8), (4, 8))
#s2 = {1: (2, 2, 3, 4), 2: (4, 5, 6, 7), 3: (7, 8, 5, 9)}
#createPlot(s1, 'xy_pair', 'line_scatter_plot', 'Title', 'Label [x]', 'Values [y]')
#createPlot(s2, 'abscissa_map', 'line_scatter_plot', 'Title', 'Label [x]', 'Values [y]')
#createPlot(s2, 'abscissa_map', 'bar_chart', 'Title', 'Label [x]', 'Values [y]')