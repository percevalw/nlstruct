import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setuptools.setup(
        name="nlstruct",
        version="0.0.5",
        author="Perceval Wajsburt",
        author_email="perceval.wajsburt@sorbonne-universite.fr",
        license='BSD 3-Clause',
        description="Natural language structuring library",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/percevalw/nlstruct",
        packages=setuptools.find_packages(),
        package_data={'': ['example.env']},
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
        install_requires=[
            "fire",
            "numpy==1.22.3",
            "torch==1.11.0",
            "unidecode==1.3.4",
            "einops==0.4.1",
            "transformers==4.30.0",
            "tqdm==4.64.0",
            "scikit-learn==1.1.0rc1",
            "pandas==1.4.2",
            "pytorch_lightning==1.4.9",
            "torchmetrics==0.7.3",
            "rich_logger==0.1.4",
            "sentencepiece==0.1.96",
            "xxhash==3.0.0",
            "regex==2020.11.13",
            "parse==1.19.0",
        ]
    )