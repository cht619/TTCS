import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle')  # Cityscapes, ACDC, Cs-foggy, CS-rainy
font_text = {'family': 'serif',
             'weight': 'normal', 'color': 'black',
             'size': 15,
             }

label_text = {'family': 'serif',
              'weight': 'normal', 'color': 'black',
              'size': 18
              }

title_text = {'family': 'serif',
              'weight': 'normal', 'color': 'black',
              'size': 18,
              }
legend_text = {
    'size': 16,
    'family': 'serif',
}


class Figures:
    def __init__(self):
        self.domain = ['Domain A', 'Domain B', 'Domain C', 'Domain D', 'Domain E']
        self.colors = ['blue', 'green', 'black', 'red', 'orange']

    def data_to_data(self, data_keys):
        # 搞错了，这里得转换成一个domain在不同params上的结果
        all_data = []
        print(data_keys.keys())
        for k in data_keys.keys():
            all_data.append(data_keys[k])
        all_data = np.array(all_data)
        return all_data.T

    # 暂定4个图？ (a) EMA alpha 因子  (b) crf map threshold (c) lora block (d) neighbour
    def plot_figure(self, axs, data, title, x_label):
        axs.set_ylabel('DSC (%)', fontdict=label_text)
        axs.set_title(title, fontdict=title_text)
        axs.grid(ls='-.', lw=0.2)
        axs.set_xticks(range(len(data.keys())))

        axs.set_xticklabels(list(data.keys()))
        axs.set_xlabel(x_label, fontdict=label_text)

        data_np = self.data_to_data(data)
        for i, d in enumerate(range(data_np.shape[0])):
            axs.plot(range(len(data_np[d])), data_np[d], linewidth=4, label=self.domain[i], color=self.colors[i], alpha=1.0,
                     marker='x', markersize=12, markerfacecolor='white')

        axs.legend(loc='best', prop=legend_text)

        # for ax in axs.flat:
        axs.tick_params(axis='x', labelsize=15)
        axs.tick_params(axis='y', labelsize=15)

    def plot(self):
        # 这里就记录dice就行，miou就忽略。这里一般说越小效果不好，distribution shift大
        ema_dice = {0.1: [32.3, 45.2, 30.3, 29.9, 42.1],
                    0.3: [35.3, 48.2, 36.3, 38.9, 43.1],
                    0.5: [50.1, 59.3, 48.2, 48.3, 53.5],
                    0.9: [66.1, 87.7, 67.2, 65.5, 74.3],
                    0.99: [65.3, 86.7, 65.2, 64.7, 76.4]}
        # 这里太大太小也不行
        threshold_dicee = {80: [62.3, 64.1, 52.3, 65.3, 72.3],
                           100: [61.3, 72.1, 70.0, 65.3, 72.3],
                           128: [65.3, 86.7, 65.2, 64.7, 76.4],
                           160: [55.3, 46.7, 35.2, 34.7, 46.4],
                           200: [20.1, 30.4, 12.2, 8.7, 19.4]}
        # sam_h有32个。这里就说3个学习不充分，5中间更好，越多效果就差
        lora_block_dice = {3: [63.5, 85.7, 62.2, 63.7, 75.4],
                           5: [65.3, 84.7, 63.2, 65.7, 76.4],
                           20: [59.3, 62.7, 64.2, 53.7, 74.4],
                           32: [58.3, 60.7, 60.2, 43.7, 75.4]
                           }

        # ORIGA数量较多，所以聚类多点。这里大概就是按这个去说，或者说越多越好.这里分析一个B(REFUGE)和A(RIM)为什么越多越差，就说噪声
        neighbour_dice = {8: [63.5, 75.0, 44.2, 63.7, 75.4],
                          16: [64.3, 86.7, 58.2, 66.7, 74.2],
                          32: [49.2, 75.7, 67.7, 66.5, 76.4]
                          }

        with PdfPages(r'./{}.pdf'.format('hyper_parameters')) as pdf:
            fig, axs = plt.subplots(2, 2, figsize=(14, 8))  # WH
            self.plot_figure(axs[0, 0], ema_dice, title='(a)', x_label=r'$\alpha$')
            self.plot_figure(axs[0, 1], threshold_dicee, title='(b)', x_label=r'threshold')
            self.plot_figure(axs[1, 0], lora_block_dice, title='(c)', x_label=r'block index')
            self.plot_figure(axs[1, 1], neighbour_dice, title='(d)', x_label=r'$U$')


            plt.tight_layout()
            pdf.savefig()
            # plt.show()
            plt.close()


# 不同SAM architecture
class Figures_bar:
    def __init__(self):
        self.RIM_ONE_r3 = [[52.4, 39.5], [66.2, 53.1], [65.3, 52.4]]
        self.REFUGE = [[70.1, 56.3], [81.6, 70.8], [86.7, 79.1]]
        self.ORIGA = [[52.9, 38.7], [68.6, 54.4], [65.2, 54.1]]
        self.REFUGE_Valid = [[48.5, 34.2], [63.7, 50.4], [64.7, 54.0]]
        self.Drishti_GS = [[49.0, 35.7], [65.6, 52.5], [71.7, 61.4]]
        self.cxr = [[4.8, 57.0], [60.2, 50.4], [65.2, 61.2]]
        self.domains = ['ViT-B', 'ViT-L', 'ViT-H']

    def plot_bar(self, ax, tick_step=1, group_gap=0.2, bar_grap=0):
        def show_bar_label(rects):
            for rect in rects:
                height = rect.get_height()
                print(rect.get_width(), height)
                ax.text(rect.get_x() + rect.get_width() / 2. - 0.10, 1.01 * height, '%s' % int(height), size=22,
                        family="serif")

        x = np.arange(len(self.domains)) * tick_step  # x每一组柱子的起点，一共是3个sub-domains
        group_nums = 3  # 多少组柱子，一组有三个
        group_width = tick_step - group_gap  # 计算一组的长度
        bar_span = group_width / group_nums  # 一组里面每个柱子的宽度
        bar_width = bar_span - bar_grap  # 是否需要bar_grap，柱子之间是否需要有空隙
        legends = ['REFUGE', 'Drishti_GS', 'X-ray']
        colors = ['black', 'red', 'blue']
        datas = [self.REFUGE, self.Drishti_GS, self.cxr]
        for i in range(3):  # 一个循环是画一个方法在所有domain上的效果，即一次是画多个间隔的柱子不是一组（相连）的柱子
            x_site = x + i * bar_span
            data = datas[i]
            data = [d[0] for d in data]
            rects = ax.bar(x_site, data, tick_label=self.domains, label=legends[i], width=bar_width, color=colors[i],
                           alpha=0.9, edgecolor='black')
            show_bar_label(rects)

        # ax.set_ylim(50, 95)
        ax.set_xticks(x + 1 * bar_span)  # 这里控制label显示的位置
        ax.set_xticklabels(self.domains, rotation=0, fontdict=label_text)
        ax.tick_params(axis='y', labelsize=22)
        ax.legend(loc='upper left', prop=legend_text)
        ax.set_ylabel('DSC (%)', fontdict=label_text)
        ax.grid()

    def plot_figure(self):
        with PdfPages(r'./{}.pdf'.format('sam_architectures')) as pdf:
            fig, axs = plt.subplots(1, 1, figsize=(12, 7))
            self.plot_bar(axs)
            plt.tight_layout()
            pdf.savefig()
            # plt.show()
            plt.close()


if __name__ == '__main__':
    # fig = Figures()
    # fig.plot()

    fig = Figures_bar()
    fig.plot_figure()