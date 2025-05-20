import cv2
import numpy as np


def save_each_figure(fig_name):
    fig = cv2.imread(fig_name)
    # cv2.imwrite('./a.png', fig[:, 1024:1024*2, :])
    # cv2.imwrite('./b.png', fig[:, 1024*2:1024*3, :])
    # cv2.imwrite('./c.png', fig[:, 1024*3:1024*4, :])
    # cv2.imwrite('./d.png', fig[:, 1024*4:, :])


    cv2.imwrite('./a.png', fig[:, :1024, :])
    cv2.imwrite('./b.png', fig[:, 1024:1024*2, :])
    cv2.imwrite('./c.png', fig[:, 1024*4:, :])
    # cv2.imwrite('./d.png', fig[:, 1024*4:, :])




if __name__ == '__main__':
    # fig_name = './Run/SAMCLIP_TTA_All/RIM_ONE_r3/Output/N-83-L.png'
    # fig_name = './Run/SAMCLIP_TTA_All/REFUGE/Output/n0050.png'
    # fig_name = './Run/SAMCLIP_TTA_All/ORIGA/Output/118.png'
    # fig_name = './Run/SAMCLIP_TTA_All/REFUGE_Valid/Output/T0108.png'
    # fig_name = './Run/SAMCLIP_TTA_All/Drishti_GS/Output/ndrishtiGS_047.png'

    # memory bank
    fig_name = './Run/SAMCLIP_TTA_All/Drishti_GS/Output/ndrishtiGS_072.png'


    save_each_figure(fig_name)