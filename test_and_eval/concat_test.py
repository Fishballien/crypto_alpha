# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:48:06 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
from pathlib import Path
import pandas as pd
import pickle
import toml
from tqdm import tqdm


from utils.dirutils import load_path_config, DirectoryProcessor
from test_and_eval.check_n_update import CheckNUpdate
from utils.logutils import FishStyleLogger


# %%
def load_test_data(factor_name, data_dir):
    df_gp = pd.read_parquet(data_dir / f'gp_{factor_name}.parquet')
    df_ic = pd.read_parquet(data_dir / f'icd_{factor_name}.parquet')
    # df_xicor = pd.read_parquet(data_dir / f'xicord_{factor_name}.parquet')
    df_hsr = pd.read_parquet(data_dir / f'hsr_{factor_name}.parquet')
    try:
        df_mmt = pd.read_parquet(data_dir / f'mmt_{factor_name}.parquet')
    except:
        pass
    with open(data_dir / f'bins_{factor_name}.pkl', 'rb') as file:
        bins_of_lag = pickle.load(file)
    try:
        with open(data_dir / f'bins_sw_{factor_name}.pkl', 'rb') as file:
            bins_of_lag_sw = pickle.load(file)
    except:
        bins_of_lag_sw = {}
        
    return {
        'gp': df_gp,
        'icd': df_ic,
        'hsr': df_hsr,
        'mmt': df_mmt,
        'bins': bins_of_lag,
        'bins_sw': bins_of_lag_sw,
        }


def save_test_data(factor_name, data_dir, data_dict):
    data_dict['gp'].to_parquet(data_dir / f'gp_{factor_name}.parquet')
    data_dict['icd'].to_parquet(data_dir / f'icd_{factor_name}.parquet')
    # 如果存在 'mmt' 键且不为空，保存该数据
    if 'mmt' in data_dict and data_dict['mmt'] is not None:
        data_dict['mmt'].to_parquet(data_dir / f'mmt_{factor_name}.parquet')
    data_dict['hsr'].to_parquet(data_dir / f'hsr_{factor_name}.parquet')

    # 保存 bins_of_lag 数据
    with open(data_dir / f'bins_{factor_name}.pkl', 'wb') as file:
        pickle.dump(data_dict['bins'], file)
    
    # 保存 bins_of_lag_sw 数据，确保 bins_sw 存在
    if 'bins_sw' in data_dict:
        with open(data_dir / f'bins_sw_{factor_name}.pkl', 'wb') as file:
            pickle.dump(data_dict['bins_sw'], file)
            

def merge_common_keys(updater, dict1, dict2, parent_key_chain=None):
    # 初始化键链
    if parent_key_chain is None:
        parent_key_chain = []
    
    merged_dict = {}
    try:
        common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
    except:
        breakpoint()

    # 遍历共有键
    for key in common_keys:
        # 更新当前键链
        current_key_chain = parent_key_chain + [str(key)]
        print(current_key_chain)

        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            # 如果两者都是字典，递归处理并传递当前的键链
            merged_dict[key] = merge_common_keys(updater, dict1[key], dict2[key], current_key_chain)
        elif isinstance(dict1[key], pd.DataFrame) and isinstance(dict2[key], pd.DataFrame):
            # 如果两者都是 DataFrame，打印当前键链并拼接
            key_chain_name = f"{' -> '.join(current_key_chain)}"
            merged_dict[key] = updater.check_n_update(key_chain_name, dict1[key], dict2[key])
        else:
            raise ValueError(f"Incompatible types at key: {' -> '.join(current_key_chain)}")

    return merged_dict


# %%
class BatchConcat:
    
    def __init__(self, batch_concat_name):
        self.batch_concat_name = batch_concat_name
        
        self._load_path_config()
        self._init_dir()
        self._load_params()
        self._init_log()
        
    def _load_path_config(self):
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]
        self.path_config = load_path_config(project_dir)
        
    def _init_dir(self):
        self.param_dir = Path(self.path_config['param']) / 'batch_test'
        self.test_dir = Path(self.path_config['result']) / 'test'
        
    def _load_params(self):
        self.params = toml.load(self.param_dir / f'{self.batch_test_name}.toml')
        
    def _init_log(self):
        self.log = FishStyleLogger()
        
    def _init_updater(self):
        check_params = self.params['check_params']
        
        self.updater = CheckNUpdate(check_params, log=self.log)
        
    def run(self):
        if 'concat_targets' in self.params:
            self._concat_by_targets()
        elif 'root_dir_dict' in self.params:
            self._concat_by_directory()
        else:
            self.log.error('Need at least one of {test_targets, root_dir_dict} to test!')

    def _concat_by_targets(self):
        raise NotImplementedError()
# =============================================================================
#         concat_targets = self.params['concat_targets']
#         
#         for test_info in concat_targets:
#             test_names = test_info['test_name']
#             test_names = [test_names] if isinstance(test_names, str) else test_names
#             for test_name in test_names:
#                 test_name_dir = self.test_dir / test_name
#                 factor_data_dir = test_info['factor_data_dir']
#                 process_name = test_info['process_name']
#                 tag_name = test_info['']
#                 kwargs = {f'{tag_type}_dir': tag_dir_mapping[tag_type] / process_name
#                           for tag_type in tag_dir_mapping}
#                 param = {pr: test_info[pr] if pr != 'test_name' else test_name for pr in test_info}
#                 param.update({'skip_plot': skip_plot, 'n_workers': n_workers})
#                 self._test_one_process(param)
# =============================================================================
                
    def _concat_by_directory(self):
        root_dir_dict = self.params['root_dir_dict']
        test_name = self.params['test_name']
        
        test_name_dir = self.test_dir / test_name
        
        dirp = DirectoryProcessor(root_dir_dict)
        mapping = dirp.mapping

        for root_dir in mapping:
            root_info = mapping[root_dir]
            tag_name = root_info['tag_name']
            tag_dir_mapping = {tag_type: test_name_dir / tag_name[tag_type] 
                               for tag_type in ('his', 'inc', 'updated')}
            process_name_list = root_info['leaf_dirs']

            self.log.info(f'Root Start: {root_dir}')
            for process_name in process_name_list:
                kwargs = {f'{tag_type}_dir': tag_dir_mapping[tag_type] / process_name
                          for tag_type in tag_dir_mapping}
                kwargs.update({'process_name': process_name, 'root_dir': root_dir})
                self._test_one_process(kwargs)
            self.log.success(f'Root Finished: {root_dir}')
                
    def _concat_one_process(self, kwargs):
        self.log.info(f"Start concating {kwargs['process_name']}")
        
        factor_dir = kwargs['root_dir'] / kwargs['process_name']
        factor_name_list = [path.stem for path in factor_dir.glob('*.parquet')]
        for factor_name in tqdm(factor_name_list, desc=kwargs['process_name']):
            his_test = load_test_data(factor_name, kwargs['his_dir'])
            inc_test = load_test_data(factor_name, kwargs['inc_dir'])
            updated_test = merge_common_keys(self.updater, his_test, inc_test)
            save_test_data(factor_name, kwargs['updated_dir'], updated_test)
        
        self.log.success(f"Finished concating {kwargs['process_name']}")
        

# %%
if __name__=='__main__':
    data_dir = Path(r'D:\crypto\multi_factor\factor_test_by_alpha\sample_data\test_concat')
    factor_name = 'n01'
    update_params = {
        'presicion': 1e-5,
        'timedelta_threshold': {
            'days': 0
            },
        }
    
    old_dir = data_dir / 'old'
    new_dir = data_dir / 'new'
    
    
    log = FishStyleLogger()
    
    
    data = {}
    data['old'] = load_test_data(factor_name, old_dir)
    data['new'] = load_test_data(factor_name, new_dir)
    
    updater = CheckNUpdate(update_params, log=log)
    merged = merge_common_keys(updater, data['old'], data['new'])