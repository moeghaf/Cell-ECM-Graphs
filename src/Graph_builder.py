from tifffile import imwrite, imread
import pandas as pd 
import numpy as np 
from tqdm import tqdm 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import os 
from sklearn.preprocessing import LabelEncoder
from glob import glob
import seaborn as sns 
import shutil
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import matplotlib.ticker as ticker
import seaborn as sns 
from matplotlib.patches import Patch
import networkx as nx 
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.stats import skew, kurtosis



class GraphBuilder:
    
    def __init__(self, full_stack_img_path, panel_path, cell_data_path, patch_size,save_folder, norm='znorm'):
        # Load IMC intensity stacks, panel, cell labels and cell segmentation masks. 
        self.full_stack_imgs = imread(full_stack_img_path)
        if norm == 'znorm':
            print('Applying z-norm')
            self.raw = self.full_stack_imgs.copy()
            normalized_stack = np.zeros_like(self.raw, dtype=np.float32)
            min_normalized_stack = np.zeros_like(self.raw, dtype=np.float32)

            for i in range(self.raw.shape[0]):
                
                self.scaler = StandardScaler()
                self.scaler_min_max = MinMaxScaler()
                normalized_stack[i] = self.scaler.fit_transform(self.raw[i])
                min_normalized_stack[i] = self.scaler_min_max.fit_transform(self.raw[i])
            
            self.full_stack_imgs = normalized_stack
            self.full_stack_imgs_min_max = min_normalized_stack # Used for plotting protein proportions instead of z-norm

        self.c, self.h, self.w = self.full_stack_imgs.shape
        self.panel = pd.read_csv(panel_path)
        self.cell_data = pd.read_csv(cell_data_path) 
        self.cell_y_str  = np.array(self.cell_data['celltype']) # May need to rename it to celltype 
        self.patch_size = patch_size
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
        
    def find_patch_pixels_using_centre(self, image, patch_centre, patch_size=10):
        mask = np.zeros_like(image, dtype=bool)
        # Calculate the coordinates of the patch region
        top_left_row = patch_centre[0] - patch_size // 2
        top_left_col = patch_centre[1] - patch_size // 2
        bottom_right_row = top_left_row + patch_size
        bottom_right_col = top_left_col + patch_size

        # Set the patch region in the mask to True
        mask[top_left_row:bottom_right_row, top_left_col:bottom_right_col] = True

        # Apply the mask to the original image
        patch_only_img = np.where(mask, image, 0) 
        return patch_only_img

    def find_nonzero_coordinates(self, image):
        nonzero_rows, nonzero_cols = np.nonzero(image)
        return np.column_stack((nonzero_rows, nonzero_cols))

    def find_neighbour_patch_coords(self, coord, coord_list, distance=10):
        within_distance_coords = []  # Add original coord for self loops 
        for c in coord_list:
            # Calculate the absolute difference between coordinates in both rows and columns
            row_diff = abs(coord[0] - c[0])
            col_diff = abs(coord[1] - c[1])
            # Check if both row and column differences are within the specified distance
            if row_diff <= distance and col_diff <= distance:
                within_distance_coords.append(c)
        return within_distance_coords
    
    def min_distance_between_patch_coords(self, set1, set2):
        distances = cdist(set1, set2)
        return np.min(distances), distances
        
    def add_cell_ecm_interactions(self, cell_ecm_dist_thres=10):

        ecm_coords_flipped = (self.patch_A_centre[1], -self.patch_A_centre[0])

        pixel_co_patch_A_flipepd = self.pixel_co_patch_A.copy()
        pixel_co_patch_A_flipepd[:,0] = self.pixel_co_patch_A[:,1]
        pixel_co_patch_A_flipepd[:,1] = -self.pixel_co_patch_A[:,0]

        patch_node = self.ecm_nodeid_to_patch_centre[(ecm_coords_flipped)]
        min_dist, dist = self.min_distance_between_patch_coords(pixel_co_patch_A_flipepd, self.cell_coords)
        
        # Check if below threshold 
        cell_ecm_dist = np.argwhere(dist < cell_ecm_dist_thres)
        
        # Add edge between cell and ecm nodes 
        if np.any(cell_ecm_dist): 
            for i in cell_ecm_dist[:,1]:
                cells_interacting_with_node_id = 'cell_' + str(i) # cells interacting with ecm node 
                self.G.add_edge(patch_node, cells_interacting_with_node_id, interaction='cell-ecm')
                
    def create_patches(self, padded_ecm_stack=None):
        if padded_ecm_stack:
            self.padded_ecm_stack
        # Get the shape of the padded image
        rows, cols, channels = self.padded_ecm_stack.shape
        
        # Calculate the number of patches along rows and columns
        num_patches_rows = rows // self.patch_size
        num_patches_cols = cols // self.patch_size
        
        # Initialize arrays to store patches and centers
        self.patches = []
        self.centers = []
        
        # Iterate over patches
        for i in range(num_patches_rows):
            for j in range(num_patches_cols):
                # Calculate patch coordinates
                start_row = i * self.patch_size
                end_row = start_row + self.patch_size
                start_col = j * self.patch_size
                end_col = start_col + self.patch_size
                
                # Extract the patch from the image
                patch = self.padded_ecm_stack[start_row:end_row, start_col:end_col, :]
                
                # Calculate patch center
                center_row = (start_row + end_row) / 2
                center_col = (start_col + end_col) / 2
                
                # Append patch and center to respective arrays
                self.patches.append(patch)
                self.centers.append((center_row, center_col))
        
        # Convert lists to numpy arrays
        self.patches = np.array(self.patches)
        self.centers = np.array(self.centers)
        
        # Reshape patches to desired format
        self.ecm_patches_rs = self.patches.reshape(-1, self.patch_size, self.patch_size, self.c)    
        self.ecm_patches_rs = self.ecm_patches_rs[:,:,:,self.ecm_mask]
        
    def load_ecm_data(self): 
        print('ECM data loaded, cell channels zeroed out.')
        self.ecm_stack = self.full_stack_imgs.copy()
        self.ecm_mask = self.panel['ecm'].eq(1).to_numpy()
        self.ecm_stack[~self.ecm_mask, :, :] = 0    
    
    def build_cell_ecm_graph(self):
        self.get_ecm_patches()
        self.load_patch_data()
        self.build_cell_cell_graph()
        self.set_up_colors()
        self.build_ecm_ecm_graph()

    def pad_ecm_stack(self):
        print('/n ----------')
        print('Padding original image for ECM patches')
        print('Original shape: ',  np.shape(self.ecm_stack))
    
        # Calculate the amount of padding required for each dimension
        pad_width = [(0, 0)] 
        pad_h = int(np.ceil(self.h/self.patch_size ) * self.patch_size )
        pad_w = int(np.ceil(self.w/self.patch_size ) * self.patch_size )
        
        for target_dim, stack_dim in zip((pad_h, pad_w), self.ecm_stack.shape[1:]):
            pad_before = (target_dim - stack_dim) // 2
            pad_after = target_dim - stack_dim - pad_before
            pad_width.append((pad_before, pad_after))
        
        # Pad the ecm_stack array
        padded_ecm_stack = np.pad(self.ecm_stack, pad_width=pad_width, mode='constant')
        # Find padded height and width 
        self.padded_ecm_stack = padded_ecm_stack.transpose(1,2,0)
        self.padded_ecm_stack_shape = np.shape(self.padded_ecm_stack[:,:,self.ecm_mask])
        print('Padded shape: ', self.padded_ecm_stack_shape)
                        

    def cluster_ecm_patches(self, k=4):

        print('Clustering ECM patches (mean intensity per channel, standard deviation, median, percentiles)')
        # Select only the 10 ECM channels 

        # Compute the mean, standard deviation, median, and percentiles for each patch
        #self.embedding = self.ecm_patches_rs.mean(axis=(1, 2))
        self.ecm_patch_flat = self.ecm_patches_rs.mean(axis=(1, 2))
        # Compute the mean, standard deviation, median, percentiles, max, min, skewness, and kurtosis for each patch
        mean_intensity = self.ecm_patches_rs.mean(axis=(1, 2))  # Mean intensity

        # Stack all features into a single array (combine across the new axis)
        self.embedding = np.stack([
               mean_intensity,
        ], axis=1)  # Combine these features as columns


        emb_dim = self.embedding.shape
        self.embedding = self.embedding.reshape(emb_dim[0], -1)
        print('embedding dim: ', self.embedding.shape)
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(self.embedding)
        
        # Save the cluster labels
        self.cluster_labels = kmeans.labels_

    def reconstruct_cluster_patches_to_image(self):
        print('Reconstructing image from cluster patches ... ')
        # Reshape patches to collapse the patch dimensions into one

        self.clustered_image = np.zeros_like(self.ecm_patches_rs)

        for i, label in enumerate(self.cluster_labels):
            self.clustered_image[i] = label    
            
        patch_size = self.ecm_patches_rs.shape[-3:]
        self.reconstructed_image = np.zeros(self.padded_ecm_stack_shape, dtype=self.ecm_patches_rs.dtype)
        
        # Counter for iterating through flattened_patches
        k = 0
        # Iterate over each block position in the original image
        for i in range(0, self.padded_ecm_stack_shape[0], patch_size[0]):
            for j in range(0, self.padded_ecm_stack_shape[1], patch_size[1]):
                for l in range(0, self.padded_ecm_stack_shape[2], patch_size[2]):
                    # Place the current patch into the appropriate position in the reconstructed image
                    self.reconstructed_image[i:i+patch_size[0], j:j+patch_size[1], l:l+patch_size[2]] = self.clustered_image[k]
                    k += 1
        
        fig, ax = plt.subplots(dpi=300)
        image = ax.imshow(self.reconstructed_image[:,:,0], cmap='jet')
        ax.grid(False)
        ax.axis('off')

        # Create a legend with cluster labels
        unique_labels = np.unique(self.cluster_labels)
        self.cluster_colors = np.array([image.cmap(image.norm(label)) for label in unique_labels])
        self.cluster_colors_map = {}
        for c, k in zip(self.cluster_colors, unique_labels):
            self.cluster_colors_map['ECM Cluster ' + str(k)] = c 

        handles = [Patch(facecolor=color, edgecolor='k', label=f'ECM Cluster {label}') for label, color in zip(unique_labels, self.cluster_colors)]
        
        # Place the legend outside the image bbox
        legend = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)
        
        for legend_handle in legend.get_children():
            if isinstance(legend_handle, plt.Line2D):
                legend_handle.set_edgecolor('black')        # Set axis labels
                legend_handle.set_linewidth(1)
        # Adjust legend and label font sizes
        for text in legend.get_texts():
            text.set_fontsize(7) 

        plt.show() 
        fig.savefig(self.save_folder+'/ECM_patch_clusters.png', bbox_inches='tight')
    
    def visualize_ecm_cluster_proteins(self):
        min_max_img = self.full_stack_imgs_min_max[self.ecm_mask].transpose(1,2,0)
        ecm_patch_img = self.reconstructed_image[:,:,0]

        cluster_means = {}

        for i in np.unique(self.cluster_labels): 
            cluster_means[i] = min_max_img[ecm_patch_img == i].mean(0)

        cluster_df = pd.DataFrame(cluster_means, index=[self.panel[self.ecm_mask].name.values])
        self.cluster_df = cluster_df.T

        #ax = self.cluster_df.plot(kind='bar', stacked=True)


        background_label = self.cluster_df.sum(axis=1).argmin()
        df = self.cluster_df.copy()

        df = df.drop(background_label)
        
        # Apply Min-Max normalisation to the remaining data to visualize it better
        df = pd.DataFrame(df, index=df.index, columns=df.columns)

        self.cluster_colors = np.delete(self.cluster_colors, background_label, axis=0)
        del self.cluster_colors_map['ECM Cluster ' + str(background_label)]

        df_percentage = df.div(df.sum(axis=1), axis=0) * 100

        df_percentage.columns = [col[0] for col in df_percentage.columns]
        # Generate HSL-based color palette with 'num_clusters' distinct colors
        colors = sns.color_palette('hls', 10)

        # Plot the bar chart with the specified colors
        ax = df_percentage.plot(kind='bar', stacked=True, color=colors, edgecolor=self.cluster_colors, 
                                linewidth=2)
        ax.grid(False)

        # Move legend outside the bounding box
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', edgecolor='black')
        for legend_handle in legend.get_children():
            if isinstance(legend_handle, plt.Line2D):
                legend_handle.set_edgecolor('black')        # Set axis labels
                legend_handle.set_linewidth(1)

        ax.set_xlabel('Extracellular matrix clusters')
        ax.set_ylabel('Protein percentage')
        # Set DPI for the figure
        plt.gcf().set_dpi(300)        
        fig = ax.get_figure()
        fig.set_dpi(300)

        # Show the plot
        plt.show()

    # Save the figure
    #fig.savefig(self.save_folder+'/ECM_protein_proportions.png', bbox_inches='tight')
    
    def get_ecm_patches(self):
        ''' Generates ECM patches '''
        self.load_ecm_data()
        self.pad_ecm_stack()
        self.create_patches()
        self.cluster_ecm_patches()
        self.reconstruct_cluster_patches_to_image()
        self.visualize_ecm_cluster_proteins()

        print('Removing background patches')
        background_idx = self.cluster_df.mean(1).argmin()
        self.background_mask = self.cluster_labels!=background_idx
        self.background_removed_ecm_patches = self.ecm_patches_rs[self.background_mask]
        self.background_removed_labels = self.cluster_labels[self.background_mask]
        
    def load_patch_data(self):


        label_encoder = LabelEncoder()
        self.cell_y = label_encoder.fit_transform(self.cell_y_str)
        

        self.ecm_x = self.background_removed_ecm_patches 
        self.ecm_y_str = np.array(['ECM Cluster ' + str(i) for i in self.background_removed_labels])
        label_encoder = LabelEncoder()
        self.ecm_y = label_encoder.fit_transform(self.ecm_y_str) + len(np.unique(self.cell_y))

        self.cell_ecm_y_str = np.hstack((self.cell_y_str, self.ecm_y_str))
        cell_binary_label = ['Cells'] * len(self.cell_y_str)
        ecm_binary_label = ['ECM'] * len(self.ecm_y_str)
        self.cell_ecm_binary_label = np.hstack((cell_binary_label, ecm_binary_label))

    def build_cell_cell_graph(self,  dist_max = 15):
        print('Adding cell nodes and edges (cell-cell interactions) ... ')
        self.G = nx.Graph()

        node_count = 0 
        # Add nodes with attributes (centroid and cell type)
        for ct, centroid_0, centroid_1 in zip(self.cell_data.celltype.values,
                                                    self.cell_data['centroid-0'].values,
                                                    self.cell_data['centroid-1'].values):
            self.G.add_node('cell_'+str(node_count),cell_type=ct, centroid=(centroid_1, -centroid_0)) #deep_features=self.cell_deep_features[node_count])
            node_count+=1
            
        def distance(centroid1, centroid2):
            return np.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
        
        # Add edges between nearest neighbors
        for node1 in self.G.nodes():
            if 'cell' in node1:  # Check if node ID contains 'cell'
                centroid1 = self.G.nodes[node1]['centroid']
                distances = [(node2, distance(centroid1, self.G.nodes[node2]['centroid'])) for node2 in self.G.nodes() if node1 != node2 and 'cell' in node2]
                distances.sort(key=lambda x: x[1])
                for neighbor, dist in distances[:5]:
                    if dist < dist_max: 
                        self.G.add_edge(node1, neighbor, distance=dist, interaction='cell-cell')

        print('Cell nodes: ', self.G.number_of_nodes())
        print('Cell-cell interactions: ', self.G.number_of_edges())               
        self.cell_G = self.G.copy()        
    
    def visualize_cell_cell_interactions(self):
        # Extracting node attributes for nodes with 'cell' in ID
        cell_nodes = [node for node in self.G.nodes() if 'cell' in node]
        node_positions = {node: attributes['centroid'] for node, attributes in self.G.nodes(data=True) if node in cell_nodes}
        node_cell_types = {node: attributes['cell_type'] for node, attributes in self.G.nodes(data=True) if node in cell_nodes}
        unique_cell_types = np.unique(list(node_cell_types.values()))
        edges_to_plot = []
        for u,v,attr in self.G.edges(data=True):
            # Check if edge is ecm-ecm interaction 
            if 'cell-cell' in attr['interaction']:
                edges_to_plot.append((u,v))
        # Generating node colors for nodes with 'cell' in ID
        node_colors = [self.color_map[node_cell_types[node]] for node in node_positions]

        # Plotting the graph with only 'cell' nodes
        fig,ax = plt.subplots(figsize=(10, 8), dpi=300)

        ax.axis('off') 
        plt.axis('off')
        nx.draw_networkx_nodes(self.G, pos=node_positions, nodelist=cell_nodes, node_color=node_colors, node_size=5, ax=ax)
        nx.draw_networkx_edges(self.G, pos=node_positions, edgelist=edges_to_plot, width=1, alpha=1, ax=ax, edge_color='darkblue')  # Draw edges only among 'cell' nodes

        # Creating legend handles for cell types
        legend_handles = []
        for cell_type, color in sorted(self.color_map.items()):
            if cell_type in unique_cell_types: 
                legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=cell_type, markersize=10, markerfacecolor=color))


        # Adding  legend
        edge_legend_handle = plt.Line2D([0], [0], color='darkblue', lw=1, alpha=1, label='Cell-Cell Interaction')
        legend_handles.append(edge_legend_handle)
        plt.legend(handles=legend_handles,  loc='upper left', bbox_to_anchor=(1.02, 1))

        plt.show()
        fig.savefig(self.save_folder+'/cell_graph.png', bbox_inches='tight')

    def set_up_colors(self):
        # Access celltype and node labels 
        self.node_labels_all = []
        self.cell_or_ecm_node = []
        for node,attr in self.G.nodes(data=True):
            if 'ecm_coords' in attr: 
                self.node_labels_all.append('ecm_cluster_' + str(attr['ecm_labels']))
                self.cell_or_ecm_node.append('ecm')
            if 'centroid' in attr: 
                self.node_labels_all.append(attr['cell_type'])
                self.cell_or_ecm_node.append('cell')
                
        
        # Add consistent node colors 
        print('Setting up consistent colors for visualization ')
        unique_nodes = list(set(self.node_labels_all))
        self.color_map = defaultdict(lambda: 'blue')  
        self.color_map.update({n: plt.cm.tab20(i) for i, n in enumerate(unique_nodes)})        
        self.node_colors = [self.color_map[node] for node in self.node_labels_all]

    def build_ecm_ecm_graph(self, min_dist_thres = 10):

        # Find patch centres, ecm labels, pixel co-ords and deep features 
        centers = np.array(self.centers)[self.background_mask]
        ecm_labels = self.cluster_labels[self.background_mask]
        ecm_coords = (centers[:, 1], -centers[:, 0])
        
        self.color_map.update(self.cluster_colors_map)
       


        # Add nodes along with attributes to the graph
        print('Adding ECM nodes ...')
        for i in range(len(ecm_labels)):
            node_id = f"ecm_node{i+1}"
            attributes = {
                'ecm_labels': ecm_labels[i],
                'ecm_coords': (ecm_coords[0][i],ecm_coords[1][i] )}
            self.G.add_node(node_id, **attributes)

        # Adding ecm-ecm-interactions
        print('Adding edges: ecm-ecm and cell-ecm interactions to graph ...')        
        # Convert edge co-ords to node id 
        self.ecm_nodeid_to_patch_centre = {}
        for i in range(np.shape(ecm_coords)[1]):
            node_id = f"ecm_node{i+1}"
            self.ecm_nodeid_to_patch_centre[ecm_coords[0][i], ecm_coords[1][i]] = node_id # centre          ecm_coords = (centers[:, 1], -centers[:, 0])

            
        self.cell_coords = []
        for i, attr in self.G.nodes(data=True):
            if 'cell' in i:
                self.cell_coords.append(attr['centroid'])
        

        self.ecm_panel = self.panel[self.ecm_mask].reset_index(drop=True)

        
        # Loop through the patches 
        self.ecm_panel = self.panel[self.ecm_mask].reset_index(drop=True)
        mean_stack = self.ecm_stack[self.ecm_mask].mean(axis=0)

        edge_list = []
        min_dist_thres = 3 
        counter = 0 
        for patch_A_centre in centers:
            self.patch_A_centre = patch_A_centre
            patch_A = self.find_patch_pixels_using_centre(mean_stack, patch_centre = patch_A_centre.astype(int)) 
            pixel_co_patch_A = self.find_nonzero_coordinates(patch_A)
            self.pixel_co_patch_A = pixel_co_patch_A
            neighbouring_patch_centres = self.find_neighbour_patch_coords(patch_A_centre, centers)
            if np.any(pixel_co_patch_A):
                self.add_cell_ecm_interactions()  
                
            for patch_B_centre in neighbouring_patch_centres:
                if np.all(patch_A_centre != patch_B_centre): 
                        patch_B = self.find_patch_pixels_using_centre(mean_stack, patch_centre =patch_B_centre.astype(int)) 
                        pixel_co_patch_B = self.find_nonzero_coordinates(patch_B)
                        if np.any(pixel_co_patch_A) and np.any(pixel_co_patch_B): 
                            min_dist_patchA_patchB, _ = self.min_distance_between_patch_coords(pixel_co_patch_A, pixel_co_patch_B)
                            if min_dist_patchA_patchB < min_dist_thres:
                                edge_list.append((patch_A_centre, patch_B_centre))


        # Add edge between node1 and node2 
        for i in range(len(edge_list)): 
            node1 = self.ecm_nodeid_to_patch_centre[edge_list[i][0][1], -edge_list[i][0][0]]
            node2 = self.ecm_nodeid_to_patch_centre[edge_list[i][1][1], -edge_list[i][1][0]]
    
    
            self.G.add_edge(node1, node2, interaction='ecm-ecm', ppi=[])      
    
    def visualize_ecm_ecm_interactions(self, edge_color='red'): 
        fig,ax = plt.subplots(figsize=(10, 8), dpi=300)
        ax.axis('off') 

        ecm_nodes = [node for node in self.G.nodes() if 'ecm' in node]
        ecm_nodes_label = ['ECM Cluster '+ str(attri['ecm_labels']) for node, attri in self.G.nodes(data=True) if 'ecm' in node]
        node_positions = {node: attr['ecm_coords'] for node, attr in self.G.nodes(data=True) if node in ecm_nodes}
        node_ecm_types = {node: 'ecm_cluster_' + str(attributes['ecm_labels']) for node, attributes in self.G.nodes(data=True) if node in ecm_nodes}

        
        
        edges_to_plot = []
        for u,v,attr in self.G.edges(data=True):
            # Check if edge is ecm-ecm interaction 
            if 'ecm-ecm' in attr['interaction'] :
                edges_to_plot.append((u,v))
        print('Edge count: ', len(edges_to_plot))
        node_colors = [self.cluster_colors_map[label] for label in ecm_nodes_label]

        nx.draw_networkx_nodes(self.G, pos=node_positions, nodelist=ecm_nodes, node_color=node_colors, node_size=5, ax=ax)
        nx.draw_networkx_edges(self.G, pos=node_positions, edgelist=edges_to_plot, width=1, alpha=1, ax=ax, edge_color='darkgreen')  # Draw edges only among 'cell' nodes


        legend_handles = []
        unique_labels = set(ecm_nodes_label)  # Assuming ecm_nodes_label contains unique labels

        for label in unique_labels:
            color = self.cluster_colors_map[str(label)]
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=str(label), markersize=10, markerfacecolor=color))

        # Adding legend
        edge_legend_handle = plt.Line2D([0], [0], color='darkgreen', lw=1, alpha=1, label='ECM-ECM Interaction')
        legend_handles.append(edge_legend_handle)

        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.show()  
        fig.savefig(self.save_folder+'/ecm_ecm_interactions.png', bbox_inches='tight')

    def visualize_cell_ecm_interactions(self):
        pos = {}
        node_y = []
        for node,attr in self.G.nodes(data=True):
            if 'ecm_coords' in attr: 
                pos[node] =attr['ecm_coords']
                node_y.append('ECM Cluster '+ str(attr['ecm_labels']))
            if 'centroid' in attr: 
                pos[node] = attr['centroid']
                node_y.append(attr['cell_type'])

        fig,ax = plt.subplots(figsize=(10, 8), dpi=300)
        ax.axis('off') 
        
        cell_cell_edges = []
        ecm_ecm_edges = []
        cell_ecm_edges = []
        
        for u,v,attr in self.G.edges(data=True):
            if 'cell-ecm' in attr['interaction']:
                cell_ecm_edges.append((u,v))

        print(len(cell_cell_edges))
        print(len(ecm_ecm_edges))
        print(len(cell_ecm_edges))
        
        node_colors = [self.color_map[str(n)] for n in node_y]

        nx.draw_networkx_nodes(self.G, pos=pos,  node_color=node_colors, node_size=5, ax=ax)
        nx.draw_networkx_edges(self.G, pos=pos, edgelist=cell_ecm_edges, width=1, alpha=1, ax=ax, edge_color='darkred') 

        
        legend_handles = []
        for cell_type, color in sorted(self.color_map.items()):
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=cell_type, markersize=10, markerfacecolor=color))
        
        cell_ecm_handle = plt.Line2D([0], [0], color='darkred', lw=1, alpha=1, label='Cell-ECM Interaction')
        legend_handles.append(cell_ecm_handle)
                                
        # Adding legend
        plt.tight_layout()
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.show()
        fig.savefig(self.save_folder+'/cell_ecm_interactions.png', bbox_inches='tight')

    def visualize_cell_ecm_graph(self, edge_color='grey'): 
        pos = {}
        node_y = []
        for node,attr in self.G.nodes(data=True):
            if 'ecm_coords' in attr: 
                pos[node] =attr['ecm_coords']
                node_y.append('ECM Cluster '+ str(attr['ecm_labels']))
            if 'centroid' in attr: 
                pos[node] = attr['centroid']
                node_y.append(attr['cell_type'])

        fig,ax = plt.subplots(figsize=(10, 8), dpi=300)
        ax.axis('off') 
        
        cell_cell_edges = []
        ecm_ecm_edges = []
        cell_ecm_edges = []
        
        for u,v,attr in self.G.edges(data=True):
            # Check if edge is ecm-ecm interaction 
            if 'cell-cell' in attr['interaction']:
                cell_cell_edges.append((u,v))
            if 'ecm-ecm' in attr['interaction']:
                ecm_ecm_edges.append((u,v))
            if 'cell-ecm' in attr['interaction']:
                cell_ecm_edges.append((u,v))

        print(len(cell_cell_edges))
        print(len(ecm_ecm_edges))
        print(len(cell_ecm_edges))
        
        node_colors = [self.color_map[str(n)] for n in node_y]

        nx.draw_networkx_nodes(self.G, pos=pos,  node_color=node_colors, node_size=5, ax=ax)
        nx.draw_networkx_edges(self.G, pos=pos, edgelist=cell_cell_edges, width=1, alpha=0.5, ax=ax, edge_color='darkblue')  #
        nx.draw_networkx_edges(self.G, pos=pos, edgelist=ecm_ecm_edges, width=1, alpha=0.5, ax=ax, edge_color='darkgreen')  #
        nx.draw_networkx_edges(self.G, pos=pos, edgelist=cell_ecm_edges, width=1, alpha=0.2, ax=ax, edge_color='darkred')  #

        
        legend_handles = []
        for cell_type, color in sorted(self.color_map.items()):
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=cell_type, markersize=10, markerfacecolor=color))

        cell_cell_handle = plt.Line2D([0], [0], color='darkblue', lw=1, alpha=1, label='Cell-Cell Interaction')
        legend_handles.append(cell_cell_handle)
        
        ecm_ecm_handle = plt.Line2D([0], [0], color='darkgreen', lw=1, alpha=1, label='ECM-ECM Interaction')
        legend_handles.append(ecm_ecm_handle)
        
        cell_ecm_handle = plt.Line2D([0], [0], color='darkred', lw=1, alpha=1, label='Cell-ECM Interaction')
        legend_handles.append(cell_ecm_handle)
                                
        # Adding legend
        plt.tight_layout()
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.show()
        fig.savefig(self.save_folder+'/cell_ecm_graph.png', bbox_inches='tight')

    def count_cells_on_patches(self,cell_mask_path):

        print('Counting cell types on each ECM cluster')
        scaler = MinMaxScaler()
        cells_per_patch = {}
        background_idx = self.cluster_df.sum(axis=1).argmin()
        for i in np.unique(self.reconstructed_image):
            if i != background_idx: 
                seg_mask = imread(cell_mask_path)
                patch_mask = self.reconstructed_image[:,:,0].copy()
                patch_mask = patch_mask == i
                #print(i)
                #plt.imshow(patch_mask)
                #plt.show()
                seg_mask[~patch_mask] = 0 
                cell_counts = self.cell_data[self.cell_data.ObjectNumber.isin(np.unique(seg_mask))].celltype.value_counts()
                
                # Convert to DataFrame for proper scaling
                cell_counts_df = cell_counts.reset_index()
                cell_counts_df.columns = ['celltype', 'count']
                
                # Scale counts and store back in dictionary
                cell_counts_df['scaled_count'] = scaler.fit_transform(cell_counts_df['count'].values.reshape(-1, 1)).flatten()
                cells_per_patch[i] = cell_counts_df.set_index('celltype')['scaled_count'].to_dict()
        
            # Visualize 
        # Convert the data into a DataFrame
        df = pd.DataFrame(cells_per_patch).fillna(0).round(2)
        df.columns = ['ECM_cluster_'+ str(int(i)) for i in list(cells_per_patch.keys())]

        # Plotting
        plt.figure(figsize=(10, 6), dpi=300)
        sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True, linewidths=0.5, fmt='g')

        # Adding labels and title
        plt.title("Cell Type on ECM Clusters Patches")
        plt.xlabel("Sample Index")
        plt.ylabel("Cell Type")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()

