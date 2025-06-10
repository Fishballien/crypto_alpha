# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:06:38 2024

@author: Xintang Zheng

"""
# %% imports
import sys
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
# eval_name = 'agg_240715_2_regarded'
eval_name = 'esmb_objv1_regarded'
model_name_list = [
    'esmb_objv1_agg_240715_2',
    'ridge_v13_agg_240715_2_1y', 
    #'ridge_v13_agg_240712_1_1y', 
    # 'ridge_v13_agg_240712_2_1y', 
    #'ridge_v13_agg_240709_1_1y', 
    # 'ridge_v13_agg_240709_2_1y',
    #'ridge_v13_agg_prc_only_1_1y',
    # 'ridge_v13_agg_prc_only_2_1y',
    # 'ridge_v13_agg_tf_only_0715_2_1y',
    'base_for_esmb_ridge_prcall_2',
    'base_for_esmb_lgbm_prcall_2',
    'esmb_objv1_agg_prc_all_2',
    #'esmb_v1_prc_all_objv1',
    
    ]
sp = '240T'
date_start = datetime(2023, 7, 1)
date_end = datetime(2024, 3, 29)


# %%
path_config_path = project_dir/ '.path_config.yaml'
with path_config_path.open('r') as file:
    path_config = yaml.safe_load(file)
    
result_dir = Path(path_config['result'])


# %%
# process_name_list = [f'{model_name}/predict' for model_name in model_name_list]
factor_name_list = ['predict']
save_dir = result_dir / 'factor_evaluation' / eval_name
save_dir.mkdir(exist_ok=True, parents=True)


# %%
# fe = FactorEvaluation(eval_name, process_name_list, factor_name_list, sp, result_dir)
# fe.eval_one_period(date_start, date_end)

gp_dict = {}
for model_name in model_name_list:
    process_name = f'{model_name}/predict'
    data_dir = result_dir / 'model' / process_name / 'data'
    try:
        df_gp = pd.read_parquet(data_dir / f'gp_predict_{model_name}.parquet')
    except:
        df_gp = pd.read_parquet(data_dir / 'gp_predict.parquet')
    df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)]
    gp_dict[process_name] = df_gp['long_short_0']
    
# Read HSR files and store in a dictionary
hsr_dict = {}
for model_name in model_name_list:
    process_name = f'{model_name}/predict'
    data_dir = result_dir / 'model' / process_name / 'data'
    try:
        df_hsr = pd.read_parquet(data_dir / f'hsr_predict_{model_name}.parquet')
    except:
        df_hsr = pd.read_parquet(data_dir / 'hsr_predict.parquet')
    df_hsr = df_hsr[(df_hsr.index >= date_start)] # & (df_hsr.index <= date_end)
    df_hsr = df_hsr.set_index((df_hsr.index.year * 100 + df_hsr.index.month) % 10000)
    hsr_dict[process_name] = df_hsr['turnover']
    
    
FONTSIZE_L1 = 20
FONTSIZE_L2 = 18
FONTSIZE_L3 = 15

title = f"Comparison regards to {eval_name}"

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(20, 20), dpi=100, layout="constrained")

# Plot cumulative returns
ax0.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
for process_name in gp_dict:
    gp = gp_dict[process_name]
    ax0.plot(gp.cumsum(), label=process_name, linewidth=3)
diff = gp_dict["merge_agg_240902_add_syh_pos4/predict"] - gp_dict["merge_agg_240902_double3m_15d_73/predict"]
ax0.plot(diff.cumsum(), label='diff', linewidth=3)

ax0.grid(linestyle=":")
ax0.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
ax0.tick_params(labelsize=FONTSIZE_L2, pad=15)

# Plot HSR data
hsr_colors = {process_name: ax0.get_lines()[i].get_color() for i, process_name in enumerate(gp_dict.keys())}
width = 0.1  # Adjust the width of the bars to be thinner
dates = df_hsr.index.unique()
date_indices = range(len(dates))
for i, process_name in enumerate(hsr_dict.keys()):
    df_hsr = hsr_dict[process_name]
    ax1.bar([x + i * width for x in date_indices], df_hsr, width=width, label=process_name, color=hsr_colors[process_name])

# Set xtick labels to actual dates
ax1.set_xticks([x + width * (len(hsr_dict) - 1) / 2 for x in date_indices])
ax1.set_xticklabels([f'{date // 100:02d}-{date % 100:02d}' for date in dates], rotation=45)


ax1.grid(linestyle=":")
ax1.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
ax1.tick_params(labelsize=FONTSIZE_L2, pad=15)

plt.savefig(save_dir / "comparison_plot.jpg", dpi=100, bbox_inches="tight")
plt.close()