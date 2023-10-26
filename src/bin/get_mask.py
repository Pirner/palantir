import cv2
from matplotlib import pyplot as plt


def main():
    im_p = r'C:\data\palantir\flooded_dataset\dataset\train\train-label-img\6477_lab.png'
    im = cv2.imread(im_p, 0)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im = im == 5
    plt.imshow(im)
    plt.show()


if __name__ == '__main__':
    main()
