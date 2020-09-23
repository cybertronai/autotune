from setuptools import setup
setup(install_requires=['torchvision',
                        'wandb',
                        'torch',
                        'scipy',
                        'numpy',
                        'attrdict',
                        'tensorboard>=1.14',  # for PyTorch logging
                        'Pillow',
                        'future',
                        'chainer',
                        'torchcontrib',
                        'tensorflow', 'matplotlib', 'pytest'  # for summary_iterator to extract events
                        ])
