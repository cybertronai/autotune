#!/usr/bin/env python
#
# Launch a single GPU instance with jupyter notebook

import argparse
import os
import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='g4box',
                    help="instance name")
parser.add_argument('--image-name', type=str,
                    default='Deep Learning AMI (Ubuntu 18.04) Version 33.0',
                    help="name of AMI to use ")
parser.add_argument('--instance-type', type=str, default='g4dn.xlarge', # 'p3.2xlarge',
                    help="type of instance")
parser.add_argument('--password',
                    default='ladybugs',
                    help='password to use for jupyter notebook')

args = parser.parse_args()
module_path = os.path.dirname(os.path.abspath(__file__))

def main():
  task = ncluster.make_task(name=args.name,
                            instance_type=args.instance_type,
                            image_name=args.image_name)

  # upload notebook config with provided password
  jupyter_config_fn = _create_jupyter_config(args.password)
  remote_config_fn = '~/.jupyter/jupyter_notebook_config.py'
  task.upload(jupyter_config_fn, remote_config_fn)

  
  remote_config_fn = '~/.jupyter/jupyter_notebook_config.json'
  task.upload(f'{module_path}/jupyter_notebook_config.json', remote_config_fn)

  # upload sample notebook and start Jupyter server
  task.run('mkdir -p /ncluster/notebooks')
  task.upload(f'{module_path}/gpubox_sample.ipynb',
              '/ncluster/notebooks/gpubox_sample.ipynb',
              dont_overwrite=True)

  task.switch_window(1)  # run in new tmux window
  task.run('conda activate pytorch_p36')
  task.run('cd /ncluster/notebooks')
  task.run('conda install -c conda-forge jupyter_nbextensions_configurator -y')
  task.run('conda install -c conda-forge jupyter_contrib_nbextensions -y')
  task.run('jupyter nbextension enable toc2/main')
  task.run('jupyter notebook', non_blocking=True)
  task.switch_window(0)
  
  task.run('conda activate pytorch_p36')
  task.run('python -c "import torch; torch.ones((3,3)).cuda()"')
  task.run('echo ready')

  print(f'Jupyter notebook will be at http://{task.public_ip}:8888')


def _create_jupyter_config(password):
  from notebook.auth import passwd
  sha = passwd(args.password)
  local_config_fn = f'{module_path}/gpubox/gpubox_jupyter_notebook_config.py'
  temp_config_fn = '/tmp/' + os.path.basename(local_config_fn)
  os.system(f'cp {local_config_fn} {temp_config_fn}')
  _replace_lines(temp_config_fn, 'c.NotebookApp.password',
                 f"c.NotebookApp.password = '{sha}'")
  return temp_config_fn


def _replace_lines(fn, startswith, new_line):
  """Replace lines starting with starts_with in fn with new_line."""
  new_lines = []
  for line in open(fn):
    if line.startswith(startswith):
      new_lines.append(new_line)
    else:
      new_lines.append(line)
  with open(fn, 'w') as f:
    f.write('\n'.join(new_lines))


if __name__ == '__main__':
  main()
