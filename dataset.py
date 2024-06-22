import numpy as np
import pandas as pd
import warnings
import rdkit.Chem as Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import gc

def fp_str_to_array(str_):
    return np.array([int(bit) for bit in list(str_)])

def calc_fps(smiles, fpgen):
    mol = Chem.MolFromSmiles(smiles)
    fp = fp_str_to_array(fpgen.GetFingerprint(mol).ToBitString())
    return fp
    

class SynergyDataset:
    
    def __init__(self,
                 input_train,
                 input_valid,
                 input_test,
                 gene_cell_df_path='data/gene_cell_df.csv',
                 meta_info_cell_df_path='data/meta_info_cell_df.csv'):
        
        self.gene_cell_df = pd.read_csv(gene_cell_df_path,
                                        index_col=0)
        
        self.meta_info_cell_df = pd.read_csv(meta_info_cell_df_path,
                                             index_col=0)
        
        self.input_train = input_train
        self.input_valid = input_valid
        self.input_test = input_test
        self.inputs = dict()
        self.inputs['train'] = input_train
        self.inputs['valid'] = input_valid
        self.inputs['test'] = input_test
        
        if not set(['Drug1', 'Drug2', 'cell_id', 'Y']).issubset(set(self.input_train.columns)):
            warnings.warn('Columns "Drug1", "Drug2", "cell_id", "Y" should be in input dataframes')
        
        self.mol_embed = dict()
        self.gene_embed = dict()
        self.gene_data = dict()
        self.gene_meta_data = dict()
        
        self.splits = dict()
        
        self.scaler = None
        self.fpgen = None
        self.pca = None
        
        self.body_zones = None
        self.gene_names = None
        
    def supported_cell_id_names(self):
        return self.meta_info_cell_df['cell_id'].tolist()
        
    def load(self, mol_embed='fp', gene_embed='pca', **params):
        
        print('Molecule embeddings creation')
        
        if mol_embed == 'fp':
            
            radius = params.get('radius', 4)
            fpSize = params.get('fpSize', 64)
            fpgen = GetMorganGenerator(radius=radius,
                                       fpSize=fpSize)
            self.fpgen = fpgen
            
            def calc_fps(smiles):
                mol = Chem.MolFromSmiles(smiles)
                fp = fp_str_to_array(self.fpgen.GetFingerprint(mol).ToBitString())
                return fp
            
        else:
            raise ValueError('Unknown molecule feature extraction method')
            
        print('Gene data extraction')
            
        self.mol_embed['train'] = pd.DataFrame()
        self.mol_embed['train']['Drug1'] = self.input_train['Drug1'].apply(calc_fps)
        self.mol_embed['train']['Drug2'] = self.input_train['Drug2'].apply(calc_fps)
        self._load_gene_data(self.input_train, 'train')
        self._load_gene_meta_data(self.input_train, 'train')
        
        self.mol_embed['valid'] = pd.DataFrame()
        self.mol_embed['valid']['Drug1'] = self.input_valid['Drug1'].apply(calc_fps)
        self.mol_embed['valid']['Drug2'] = self.input_valid['Drug2'].apply(calc_fps)
        self._load_gene_data(self.input_valid, 'valid')
        self._load_gene_meta_data(self.input_valid, 'valid')
        
        self.mol_embed['test'] = pd.DataFrame()
        self.mol_embed['test']['Drug1'] = self.input_test['Drug1'].apply(calc_fps)
        self.mol_embed['test']['Drug2'] = self.input_test['Drug2'].apply(calc_fps)
        self._load_gene_data(self.input_test, 'test')
        self._load_gene_meta_data(self.input_test, 'test')
        
        if gene_embed == 'pca':
            print('Gene PCA features creation')
            gene_names = self.gene_data['train'].columns.tolist()[4:]
            self.gene_names = gene_names
            
            genes_train = self.gene_data['train'][gene_names]
            genes_valid = self.gene_data['valid'][gene_names]
            genes_test = self.gene_data['test'][gene_names]
            
            scaler = MinMaxScaler()
            self.scaler = scaler
            genes_scaled_train = self.scaler.fit_transform(genes_train)
            genes_scaled_valid = self.scaler.transform(genes_valid)
            genes_scaled_test = self.scaler.transform(genes_test)
            
            pca = PCA(n_components=params.get('n_components', 10))
            self.pca = pca
            gene_features_train = self.pca.fit_transform(genes_scaled_train)
            gene_features_valid = self.pca.transform(genes_scaled_valid)
            gene_features_test = self.pca.transform(genes_scaled_test)
            
        else:
            raise ValueError('Unknown gene expression encoding method')
            
        print('Cell body zone information encoding')
        enc = OneHotEncoder(handle_unknown='ignore')
        X = self.gene_meta_data['train'][['body_zone']]
        enc.fit(X)
        self.body_zones = enc.categories_[0].tolist()
        
        df_train = pd.DataFrame()
        df_train = self.mol_embed['train']
        df_train['pca_features'] = gene_features_train.tolist()
        df_train['body_zone'] = enc.transform(X).toarray().tolist() 
        self.splits['train'] = df_train
        
        df_valid = pd.DataFrame()
        df_valid = self.mol_embed['valid']
        df_valid['pca_features'] = gene_features_valid.tolist()
        df_valid['body_zone'] = enc.transform(
            self.gene_meta_data['valid'][['body_zone']]
        ).toarray().tolist()
        self.splits['valid'] = df_valid
        
        df_test = pd.DataFrame()
        df_test = self.mol_embed['test']
        df_test['pca_features'] = gene_features_test.tolist()
        df_test['body_zone'] = enc.transform(
            self.gene_meta_data['test'][['body_zone']]
        ).toarray().tolist()
        self.splits['test'] = df_test
      
    def _load_gene_data(self, df, split_type):
        
        gene_data = df.copy()
        result = pd.merge(gene_data,
                          self.gene_cell_df,
                          how="left",
                          on='cell_id')
        self.gene_data[split_type] = result
        
        del gene_data 
        gc.collect()
    
    def _load_gene_meta_data(self, df, split_type):
        
        gene_data = df.copy()
        result = pd.merge(gene_data,
                          self.meta_info_cell_df,
                          how="left",
                          on='cell_id')
        self.gene_meta_data[split_type] = result
        
        del gene_data 
        gc.collect()
    
    def get_item(self, idx, split_type='train'):
        
        smi1 = self.inputs[split_type].iloc[idx]['Drug1']
        smi2 = self.inputs[split_type].iloc[idx]['Drug2']
        cell_id = self.inputs[split_type].iloc[idx]['cell_id']
        y = self.inputs[split_type].iloc[idx]['Y']
        
        return smi1, smi2, cell_id, self.splits[split_type].iloc[idx], y 
        
        
        
        