import os
import yaml


def make_yaml(pl_model, dataset, check_pointer_path):
    if not os.path.exists(check_pointer_path):
        os.mkdir(check_pointer_path)
    d_opt = pl_model.configure_optimizers()
    d_opt['optimizer'] = str(d_opt['optimizer']).split('\n')
    d_opt['lr_scheduler'] = str(d_opt['lr_scheduler']).split('\n')
    meta_data = {**d_opt, **dataset}
    meta_data_path = os.path.join(check_pointer_path, 'meta_data.yml')
    with open(meta_data_path, 'w') as outfile:
        yaml.dump(meta_data, outfile, default_flow_style=False)
