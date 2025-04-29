# workflow_16s


<pre> 
  workflow_16s/ 
  ├── references/ 
  │ ├── classifier/ 
  │ │ └── silva-138-99-515-806/ 
  │ ├── conda_envs/
  │ ├── manual_metadata/ 
  │ ├── config.yaml
  │ ├── datasets.tsv
  │ └── datasets.txt
  ├── src/ 
  │ ├── workflow_16s/
  │ │ ├── ena/
  │ │ ├── figures/
  │ │ ├── metadata/
  │ │ ├── qiime/
  │ │ ├── sequences/
  │ │ ├── utils/
  │ │ ├── __init__.py 
  │ │ ├── config.py 
  │ │ └── logger.py 
  │ ├── __init__.py 
  │ └── run.py 
  ├── README.md
  └── setup.sh 
  </pre>

# Config file:

### ⟦ ENA ⟧
- **email**: Your email used to login to ENA.
  
### ⟦ Dataset List ⟧
- **Default**: ```"./datasets.txt"```
  
### ⟦ Dataset Information ⟧
- **Default**: ```"./datasets.tsv"```
  
### ⟦ Project Directory ⟧
- **Default**: ``````
  
### ⟦ ⟧
- **Default**: ``````
  
### ⟦ ⟧
- **Default**: ``````
  
### ⟦ ⟧
- **Default**: ``````
  

# Known issues:
- Datasets with SingleEnd sequences fail per-dataset QIIME workflow.
