from setuptools import setup, find_packages

setup(
    name="keras_reservoir_computing",
    version="0.1.0",
    description="A package for training and using Echo State Networks and Reservoir Computing in general.",
    author="Daniel Estevez",
    author_email="kemossabee@gmail.com",
    url="https://github.com/ZentropyUH/ESN",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "keras==3.8.0",
        "networkx==3.4.2",
        "pandas==2.2.3",
        "scipy==1.14.1",
        "tensorflow==2.18.0",
        "rich==13.9.4",
        "matplotlib==3.9.2",
        "numpy==2.0.2",
        "ipykernel==6.29.5",
        "ipywidgets==8.1.5",
        "ipympl==0.9.4",
        "netCDF4==1.7.2",
        "xarray==2025.1.1"

    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
