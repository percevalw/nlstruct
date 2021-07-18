import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setuptools.setup(
        name="pyner",
        version="0.0.1",
        author="Perceval Wajsburt",
        author_email="perceval.wajsburt@sorbonne-universite.fr",
        license='BSD 3-Clause',
        description="Named entity recognition library",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/percevalw/pyner",
        packages=setuptools.find_packages(),
        package_data={},
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
        install_requires=[
            "numpy~=1.19.5",
            "torch~=1.7.1",
            "unidecode~=1.1.2",
            "einops~=0.3.0",
            "transformers~=4.3.0",
            "optuna~=2.5.0",
            "tqdm~=4.56.0",
            "sklearn~=0.0",
            "scikit-learn~=0.24.1",
            "pandas~=1.2.1",
            "pytorch_lightning~=1.1.7",
            "rich_logger~=0.1.3",
            "sentencepiece~=0.1.95",
        ]
    )
