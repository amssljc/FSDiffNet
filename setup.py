from setuptools import setup, find_packages

setup(
    name='fsdiffnet',
    version='0.12',
    author='Jiacheng Leng',
    author_email='jcleng@zhejianglab.com',
    packages=find_packages(),
    package_data={
        'fsdiffnet': [
            'models/*',
        ],
    },
    install_requires=[
        'matplotlib',
        'nilearn',
        'networkx',
        'numpy',
        'pandas',
        'progressbar2', 
        'rpy2',
        'scikit-learn',
        'scipy',
        'seaborn',
        'torch',
        'torchaudio',
        'torchvision',
        'tqdm',
        'wandb'
    ],
    include_package_data=True,
    zip_safe=False,
)
