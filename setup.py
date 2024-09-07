from setuptools import setup, find_packages

setup(
    name='SelectiveUpdateGru',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    author='ezpea',
    author_email='ezpea',
    description='A custom GRU that selectively updates its hidden state based on a control signal.',
    url='https://github.com/ezpea/SelectiveUpdateGru',
)

