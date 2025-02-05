# Cell-ECM-Graphs in Imaging Mass Cytometry

As an extension of traditional cell graphs, we present a novel framework called cell-ECM graphs, which incorporates both cellular and ECM components into a single graph, allowing for the analysis of cell-cell, ECM-ECM, and cell-ECM interactions. 

## Prerequisites Checklist ✅
This pipeline requires the use of Steinbock pipeline for cell segmentation and cell annotation. The path to these files/folder will need to be used in step 6.

- [ ] panel.csv - panel of markers with an 'ecm' column, with ECM markers as 1.

- [ ] imgs - Folder containing full stack images (NxHxW), where N is the number of markers, can be raw or preprocessed.

- [ ] cell_data - Folder containing should contain cell data ('centroid-0','centroid-1' and 'celltype') per ROI. 

![Method Overview](Figure_1.png)

## Requirments
Python 3.12.3

`git` for cloning the repository

## Usage Examples
### Python Notebook: 
For a more customizable and Python-focused experience, check out the [Lung_single ROI](tutorial/Lung_single_ROI.ipynb)
and [Lung_multiple ROI](tutorial/Lung_multiple_ROI.ipynb) tutorials.

### Command Line:  
Can be used to process multiple ROIs easily:



#### Step 1: Clone Cell-ECM-Graphs
```
git clone https://github.com/moeghaf/Cell-ECM-Graphs.git
```

#### Step 2: Navigate to Cell-ECM-Graphs
```
cd Cell-ECM-Graphs
```


#### Step 3: Create new environment and activate 
```
conda create --name cellECMgraphs python=3.11.9

conda activate cellECMgraphs
```


#### Step 4: Install requirements 
```
pip install -r requirements.txt
```


#### Step 5: Change dir to source code 
```
cd src 
```

#### Step 6: Run CellECMGraphs_CLI using the above  
```
python CellECMGraphs_CLI.py --img_dir C:/PATH/TO/YOUR/imgs --panel_path C:/PATH/TO/YOUR/panel.csv --cellData_dir C:/PATH/TO/YOUR/cell_data
```






