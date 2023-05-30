import setuptools

with open("README.md", 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='keypoint-sort',
    version='0.0.0',
    author='Caleb Weinreb',
    author_email='calebsw@gmail.com',
    description='Model-based keypoint sorting for social pose tracking',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.8',
    install_requires=[
        'jax>=0.3',
        'networkx',
    ], 
    url='https://github.com/calebweinreb/keypoint-sort'
)