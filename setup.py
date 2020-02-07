import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nlstruct",  # Replace with your own username
    version="0.0.1",
    author="Perceval Wajsburt",
    author_email="perceval.wajsburt@sorbonne-universite.fr",
    description="Natural language structuring library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/perceval/nlstruct",
    packages=setuptools.find_packages(),
    package_data={'': ['LICENSE', 'README', 'example.env']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
