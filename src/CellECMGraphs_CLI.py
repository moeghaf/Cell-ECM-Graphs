### Command Line Useage of Cell-ECM graphs ### 

import argparse
import random
import os 
import sys
sys.path.append(os.path.abspath("../src")) 
from Graph_builder import * 
from permutation_test import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import random
from CellECMGraphs_multiple import *

random.seed(42)  
np.random.seed(42)  

# Start up logo
logo = r"""
===========================================================================
  ____     _ _       _____ ____ __  __    ____                 _         
 / ___|___| | |     | ____/ ___|  \/  |  / ___|_ __ __ _ _ __ | |__  ___ 
| |   / _ \ | |_____|  _|| |   | |\/| | | |  _| '__/ _` | '_ \| '_ \/ __|
| |__|  __/ | |_____| |__| |___| |  | | | |_| | | | (_| | |_) | | | \__ \
 \____\___|_|_|     |_____\____|_|  |_|  \____|_|  \__,_| .__/|_| |_|___/
                                                        |_|                                                                     
  Integrating the Extracellular Matrix into Cell Graphs
===========================================================================
"""
print('\n ')

print(logo)
print("🔬 Starting Cell-ECM-Graphs (CEG)...\n")


# Params for Cell ECM Graphs 
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, help='path to full stack raw images', required=True)
parser.add_argument('--panel_path', type=str, help='path to panel, requires ecm column, with ECM markers as 1', required=True)
parser.add_argument('--cellData_dir', type=str, help='path to cell data (cell types, objectID linking to masks)', required=True)
parser.add_argument('--cell_masks', type=str, help='path to cell segmentation masks from Steinbock', required=False)
parser.add_argument('--patch_size', type=int, help='ECM patch size, default is 10', default=10)
parser.add_argument('--cell_KNN', type=int, help='Number of cell neighbours', default=5)
parser.add_argument('--cell_distance', type=int, help='Minimum distance between cells to be interacting', default=15)
parser.add_argument('--ECM_distance', type=int, help='Minimum distance between ECM active pixels to be interacting', default=3)
parser.add_argument('--cell_ECM_distance', type=int, help='Minimum distance between cell and ECM active pixels to be interacting', default=10)
parser.add_argument('--norm', type=str, help='Channel-wise normalized - minmax, znorm, or none', default='znorm')
parser.add_argument('--save_folder', type=str, help='Location to save results', default='Results')
args = parser.parse_args()

print('\n----\n')
print('TESTING WITH ONLY 2')
print('\n----\n')

full_stack_img_dir = np.sort(glob(args.img_dir + '*/img/*'))[:2] 
cell_data_dir = np.sort(glob(args.cellData_dir + '/*'))[:2] 


ceg = Cell_ECM_Graphs(full_stack_img_path=full_stack_img_dir, 
                      panel_path=args.panel_path,
                      cell_data_path=cell_data_dir,
                      norm=args.norm,
                      save_folder=args.save_folder)


''''''
print('Initiated CEG')
print('\n----\n')
ceg.prechecks()
print('\n----\n')
ceg.build_multiple_graphs()
print('\n----\n')
ceg.joint_ecm_clustering()
print('\n----\n')
ceg.visualize_joint_ecm_clustered_patches()
print('\n----\n')
ceg.visualize_cluster_protein_percentages
print('\n----\n')
ceg.visualize_multiple_graphs()
print('\n---------')
print('Finished.')
print('Check Results folder for images.')
print('---------')
