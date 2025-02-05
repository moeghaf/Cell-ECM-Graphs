

### Imports ### 

import pandas as pd 
from tifffile import imread, imwrite 
import numpy as np 
import os 
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from permutation_test import *
from Graph_builder import GraphBuilder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import argparse
from matplotlib.patches import Patch



### Cell Matrix Graph Main Module  ###

class Cell_ECM_Graphs(GraphBuilder):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Pass manually provided arguments
        self.single=False
    
    def prechecks(self):

        # Check panel has ecm column
        self.panel = pd.read_csv(self.panel_path)
        if 'ecm' in self.panel.columns:
            print('Check list: 0/2')
            print(' - ecm column found in panel. 1/2')
        else:
            raise ValueError('Failed check, please add ecm column to panel, with 1 indicating ECM markers')
        
        # Check if celltype in cell data column
        celltype_bool = []
        for i in self.cell_data_path:
            df = pd.read_csv(i)
            if 'celltype' in df:
                celltype_bool.append(True)
            else:
                celltype_bool.append(False)    
        if np.all(celltype_bool):
            print(' - celltype found in cell data files. 2/2')
        else:
            raise ValueError('Failed checks, please add celltype column to celldata file' )
    
    def build_multiple_graphs(self):
        ''' To process multiple ROIs together '''
        self.ceg_dict = {}
        print('Building Cell-ECM-Graphs...')
        for i in range(len(self.full_stack_img_path)):

            save_folder_sub = self.save_folder + '/Results' + '_' + str(i)
            os.makedirs(save_folder_sub, exist_ok=True)
            ceg =  GraphBuilder(full_stack_img_path=self.full_stack_img_path[i],
                                panel_path=self.panel_path,
                                cell_data_path=self.cell_data_path[i],
                                patch_size=self.patch_size, 
                                save_folder = save_folder_sub,  
                                norm=self.norm,
                                single=False
                                )
            ceg.load_imgs()
            ceg.build_cell_ecm_graph()
            self.ceg_dict[i] = ceg
            print('ROI '+str(i)+ ' complete.')   

    def joint_ecm_clustering(self):
        print('Clustering all ECM patches together ... ')
        # get ECM patches 
        means = []
        patches_per_image = self.ceg_dict[0].ecm_patches_rs.shape[0] 
        for i in range(len(self.ceg_dict)):
            means.append(self.ceg_dict[i].ecm_patches_rs.mean(axis=(1,2)))
        ecm_means = np.concatenate(means)

        # Recluster 
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=30, algorithm='elkan')
        kmeans.fit(ecm_means)
        cluster_labels = kmeans.labels_
        
        # Find background label 
        background_means = []
        for i in np.unique(cluster_labels):
            mask = cluster_labels == i 
            background_means.append(ecm_means[mask].mean())
        background_idx = np.argmin(background_means)
        
        # Update cluster labels and background 
        cluster_labels_per_G = {}
        background_per_G = {}
        counter = 0 
        
        for i in self.ceg_dict:
            cluster_labels_per_G[i] = cluster_labels[(patches_per_image*counter):(patches_per_image*(counter+1))]
            background_per_G[i] = cluster_labels_per_G[i] != np.unique(cluster_labels)[background_idx]
            counter +=1 
            self.ceg_dict[i].cluster_labels = cluster_labels_per_G[i]
            self.ceg_dict[i].background_mask = background_per_G[i]
            ecm_nodes_to_remove = [n for n,attri in self.ceg_dict[i].G.nodes(data=True) if 'ecm' in n ]
            self.ceg_dict[i].G.remove_nodes_from(ecm_nodes_to_remove)

            edges_to_remove = [(u, v) for u, v, attri in self.ceg_dict[i].G.edges(data=True) if attri['interaction'] == 'cell-ecm']
            self.ceg_dict[i].G.remove_edges_from(edges_to_remove)
        
            edges_to_remove = [(u, v) for u, v, attri in self.ceg_dict[i].G.edges(data=True) if attri['interaction'] == 'ecm-ecm']
            self.ceg_dict[i].G.remove_edges_from(edges_to_remove)
            self.ceg_dict[i].build_ecm_ecm_graph()
            self.ceg_dict[i].reconstruct_cluster_patches_to_image()
        
        self.ceg_dict[0].set_up_colors()
        for i in range(len(self.ceg_dict)):
            self.ceg_dict[i].color_map = self.ceg_dict[0].color_map       

    def visualize_joint_ecm_clustered_patches(self):
        print('Visualizing joint ECM clusters ... ')
        for i in range(len(self.ceg_dict)):
            fig, ax = plt.subplots(dpi=300)
            if self.single == False:
                plt.close(fig)
            image = ax.imshow(self.ceg_dict[i].reconstructed_image[:,:,0], cmap='jet')

            ax.grid(self.ceg_dict[i])
            ax.axis('off')

            # Create a legend with cluster labels
            unique_labels = np.unique(self.ceg_dict[i].cluster_labels)
            self.ceg_dict[i].cluster_colors = np.array([image.cmap(image.norm(label)) for label in unique_labels])
            self.ceg_dict[i].cluster_colors_map = {}
            for c, k in zip(self.ceg_dict[i].cluster_colors, unique_labels):
                self.ceg_dict[i].cluster_colors_map['ECM Cluster ' + str(k)] = c 

            handles = [Patch(facecolor=color, edgecolor='k', label=f'ECM Cluster {label}') for label, color in zip(unique_labels, self.ceg_dict[i].cluster_colors)]

            # Place the legend outside the image bbox
            legend = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)

            for legend_handle in legend.get_children():
                if isinstance(legend_handle, plt.Line2D):
                    legend_handle.set_edgecolor('black')        # Set axis labels
                    legend_handle.set_linewidth(1)
            # Adjust legend and label font sizes
            for text in legend.get_texts():
                text.set_fontsize(7) 

            fig.savefig(self.ceg_dict[i].save_folder+'/ecm_patch_clusters.png', bbox_inches='tight')
            #plt.show() 

    def visualize_multiple_graphs(self):
        print('Visualizing cell-cell, ecm-ecm, cell-ecm interactions')
        # Improve method for color alignments 
        self.ceg_dict[0].color_map['ECM Cluster 1'] = self.ceg_dict[0].color_map.pop('ecm_cluster_1')
        self.ceg_dict[0].color_map['ECM Cluster 2'] = self.ceg_dict[0].color_map.pop('ecm_cluster_2')
        self.ceg_dict[0].color_map['ECM Cluster 3'] = self.ceg_dict[0].color_map.pop('ecm_cluster_3')

        for i in range(len(self.ceg_dict)):
                self.ceg_dict[i].visualize_cell_cell_interactions()
                self.ceg_dict[i].visualize_ecm_ecm_interactions()
                self.ceg_dict[i].visualize_cell_ecm_interactions()
                self.ceg_dict[i].visualize_cell_ecm_graph()    

    def visualize_cluster_protein_percentages(self):
        'Visualizing ECM proteins per ECM cluster'
        cluster_dfs = []
        for i in range(len(self.ceg_dict)):
            min_max_img = self.ceg_dict[i].full_stack_imgs_min_max[self.ceg_dict[i].ecm_mask].transpose(1,2,0)
            ecm_patch_img = self.ceg_dict[i].reconstructed_image[:,:,0]

            cluster_means = {}

            for j in np.unique(self.ceg_dict[i].cluster_labels): 

                cluster_means[j] = min_max_img[ecm_patch_img == j].mean(0)

            cluster_df = pd.DataFrame(cluster_means, index=[self.panel[self.ceg_dict[i].ecm_mask].name.values])
            background_label = cluster_df.mean(axis=0).argmin()
            df = cluster_df.copy()
            df = df.drop(background_label, axis=1)
            df_percentage = df.div(df.sum(axis=0), axis=1) * 100



            cluster_dfs.append(df_percentage)
        
        concat_dfs = pd.concat(cluster_dfs)
        df_to_plot = concat_dfs.groupby(concat_dfs.index).mean()
        colors = sns.color_palette('hls', 10)
        df_to_plot.index = [i[0] for i in df_to_plot.index]
        df_copy = df_to_plot.copy()

        # Plot the bar chart with the specified colors
        ax = df_to_plot.T.plot(kind='bar', stacked=True, color=colors, edgecolor='black',
                                linewidth=1)
        ax.grid(False)

        # Move legend outside the bounding box
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', edgecolor='black')
        for legend_handle in legend.get_children():
            if isinstance(legend_handle, plt.Line2D):
                legend_handle.set_edgecolor('black')        # Set axis labels
                legend_handle.set_linewidth(0.5)

        ax.set_xlabel('Extracellular matrix clusters')
        ax.set_ylabel('Protein percentage')
        # Set DPI for the figure
        plt.gcf().set_dpi(300)        
        fig = ax.get_figure()
        fig.set_dpi(300)

        # Show the plot
        fig.savefig(self.save_folder+'/joint_cluster_protein_exp.png', bbox_inches='tight')
        plt.close(fig)

    def test_interactions(self, cell_or_ecm):
        
        count_interactions_dict = {}
        classic_interactions_dict = {}
        sigval_dict = {}

        for k in self.ceg_dict:
            df = graph_to_df( self.ceg_dict[k])
            if cell_or_ecm != 'cell_ecm': 
                cell_df = df[df['cell_or_ecm'] == cell_or_ecm].reset_index(drop=True)
                count_interactions = count_interactions_from_df(cell_df)

            else:
                cell_df = df 
                count_interactions = count_interactions_from_df(cell_df)
                count_interactions= count_interactions.iloc[:-3,-3:]

            if cell_or_ecm == 'ecm': 
                rename_map = {'ecm_0': 'ECM Cluster 0',
                'ecm_1': 'ECM Cluster 1',
                'ecm_2': 'ECM Cluster 2',
                'ecm_3' :'ECM Cluster 3'}

                old_names = list(count_interactions.columns)
                new_names = [rename_map[i] for i in old_names]

                count_interactions.columns = new_names
                count_interactions.index = new_names

            count_interactions_dict[k] = count_interactions
            ct = classic_interaction_count( self.ceg_dict[k], count_interactions, cell_or_ecm=cell_or_ecm)
            classic_interactions_dict[k] = ct 
            pertubations = []

            for _ in range(1000): 
                if cell_or_ecm != 'both': 
                    permuted_cell_df = permute_labels_in_df(cell_df)
                    count_interactions = count_interactions_from_df(permuted_cell_df)
                else: 
                    permuted_cell_df = permute_cell_ecm_labels_in_df(cell_df)
                    count_interactions = count_interactions_from_df(permuted_cell_df)
                    count_interactions = count_interactions.iloc[:-3,-3:]


                if cell_or_ecm == 'ecm': 
                    rename_map = {'ecm_0': 'ECM Cluster 0',
                    'ecm_1': 'ECM Cluster 1',
                    'ecm_2': 'ECM Cluster 2',
                    'ecm_3' :'ECM Cluster 3'}

                    old_names = list(count_interactions.columns)
                    new_names = [rename_map[i] for i in old_names]

                    count_interactions.columns = new_names
                    count_interactions.index = new_names
                permuted_ct = classic_interaction_count( self.ceg_dict[k], count_interactions, cell_or_ecm)

                pertubations.append(permuted_ct)
                
            pertubations = np.array(pertubations)


            sigval_matrix = np.zeros_like(ct)
            n_pertubations,r,c = pertubations.shape
            p_thresh = 0.01

            for i in range(r):
                for j in range(c): 


                    p_gt = np.mean(pertubations[:,i,j] > ct.iloc[i,j]) / (n_pertubations+1)
                    p_lt = np.mean(pertubations[:,i,j] < ct.iloc[i,j]) / (n_pertubations+1)

                    interaction = p_lt > p_gt

                    if p_gt < p_lt:
                        p = p_gt 
                    else:
                        p = p_lt 
                        
                    sig = p < p_thresh

                    if interaction == False and sig == True: 
                        sigval = -1 
                    if sig == False: 
                        sigval = 0 
                    if interaction == True and sig == True:
                        sigval= 1
                    
                    sigval_matrix[i,j] = sigval
            if cell_or_ecm != 'cell_ecm':
                names = ct.columns
                sigval_matrix = pd.DataFrame((sigval_matrix), columns=names, index=names)
            else: 
                col_names = permuted_ct.columns
                row_names = permuted_ct.index
                rename_map = {'ecm_0': 'ECM Cluster 0',
                    'ecm_1': 'ECM Cluster 1',
                    'ecm_2': 'ECM Cluster 2',
                    'ecm_3' :'ECM Cluster 3'}

                old_names = list(col_names)
                new_names = [rename_map[i] for i in old_names]
                sigval_matrix = pd.DataFrame((sigval_matrix), columns=new_names, index=row_names)
                plt.figure(figsize=(10, 8))
                sns.heatmap(sigval_matrix, cmap='bwr')
                plt.show()
            sigval_dict[k] = sigval_matrix   

        conditions = np.array(self.conditions)
        sigval_keys = np.array(list(sigval_dict.keys()))

        df1  = sigval_dict[sigval_keys[conditions == 'PBS'][0]]
        count = 0 
        for i in sigval_keys[conditions == 'PBS'][1:]:
            df2 = sigval_dict[i]
            

            if count == 0 : 
                common_index = df1.index.union(df2.index)
                common_columns = df1.columns.union(df2.columns)
                df1_filled = df1.reindex(index=common_index, columns=common_columns, fill_value=0)
                df2_filled = df2.reindex(index=common_index, columns=common_columns, fill_value=0)
                result = df1_filled.copy()
                result+= df2_filled 
                count+=1 
            else: 
                common_index = result.index.union(df2.index)
                common_columns = result.columns.union(df2.columns)
                df2_filled = df2.reindex(index=common_index, columns=common_columns, fill_value=0)
                result+=df2_filled
                
            
        os.makedirs('permutation_test_results', exist_ok=True)
        plt.figure(figsize=(10, 8),dpi=300)
        sns.heatmap(result, cmap='bwr')
        if cell_or_ecm == 'cell': 
            plt.title('Significant PBS Cell-Cell Interactions ')
        if cell_or_ecm == 'ecm': 
            plt.title('Significant PBS ECM-ECM Interactions ')
        if cell_or_ecm == 'both': 
            plt.title('Significant PBS Cell-ECM Interactions ')
        plt.savefig(f'permutation_test_results/significant_PBS_interactions_{cell_or_ecm}.png', bbox_inches='tight')
        plt.show()

        df1  = sigval_dict[sigval_keys[conditions != 'PBS'][0]]
        count = 0 
        for i in sigval_keys[conditions != 'PBS'][1:]:
            df2 = sigval_dict[i]
            

            if count == 0 : 
                common_index = df1.index.union(df2.index)
                common_columns = df1.columns.union(df2.columns)
                df1_filled = df1.reindex(index=common_index, columns=common_columns, fill_value=0)
                df2_filled = df2.reindex(index=common_index, columns=common_columns, fill_value=0)
                result = df1_filled.copy()
                result+= df2_filled 
                count+=1 
            else: 
                common_index = result.index.union(df2.index)
                common_columns = result.columns.union(df2.columns)
                df2_filled = df2.reindex(index=common_index, columns=common_columns, fill_value=0)
                result+=df2_filled
                

        plt.figure(figsize=(10, 8))
        sns.heatmap(result, cmap='bwr')
        if cell_or_ecm == 'cell': 
            plt.title('Significant DRA Cell-Cell Interactions ')
        if cell_or_ecm == 'ecm': 
            plt.title('Significant DRA ECM-ECM Interactions ')
        if cell_or_ecm == 'both': 
            plt.title('Significant DRA Cell-ECM Interactions ')
        plt.savefig(f'permutation_test_results/significant_DRA_interactions_{cell_or_ecm}.png', bbox_inches='tight')

        plt.show()