import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setuptools.setup(
        name="nlstruct",  # Replace with your own username
        version="0.0.2",
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
            'numpy>=1.17.4',
            'pandas>=0.24.1',
            'pathos>=0.2.5',
            'python-dotenv>=0.10.3',
            'PyYAML>=5.2',
            'scikit-learn>=0.22',
            'scipy>=1.4.1',
            'sh>=1.12.14',
            'spacy>=2.2.3',
            'sympy>=1.5',
            'termcolor>=1.1.0',
            'torch>=1.3.1',
            'tqdm>=4.40.2',
            'xxhash>=1.4.3',
            'unidecode',
        ]
    )
