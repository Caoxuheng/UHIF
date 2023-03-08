import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path',type=str,default='E:\Multispectral Image Dataset\CAVE\Mat_dataset/', help='where you store your HSI data file')
parser.add_argument('--srf_path',type=str,default='P_N.mat', help='where you save your camera response function')
args=parser.parse_args()
