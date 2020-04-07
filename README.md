## Features
#### No magic, no config files
Every preprocessing, training and prediction can be written from scratch in a few lines
without ever having to take a deep dive into a list of complex prewritten functions, and magically injected dependencies.

Since every problem needs a different pipeline, being able to quickly write those functions from scratch can be highly beneficial.

#### Almost no predefined structures
Jupyter has become the ML community main tool to analyze data and build predictive models. However, it does not combine well with object oriented programming, since one of Jupyter main advantages (in my opinion) is to be able to execute your code step by step.

Libraries that rely on precoded methods and functions for processing data thus quickly become burdersome when they don't fit one's needs. Once again, this library allows you to write your own code quickly, without having to rely on some restrictive objects and, later, package your model as you want. 

#### Pandas-based preprocessing
Most of the input or infered data can be expressed as a DataFrame of features, ids and span indices.

This library therefore takes advantage of pandas advanced frame indexing and combining features to make the preprocessing short, easy and explicit.

#### Easy nested/relational batching
In structuring problems (and for text data all the more), features can be highly relational. 
Other libraries rely on predefined nested structures that don't always fit one's needs.

This library introduce a flexible, yet performant batching structure that allows the user
to switch between numpy, scipy and torch matrices and easily split relational data.
Train/dev splits or minibatching follow the same indexing patterns.

For example, this structure enables an efficient character encoder implementation in just a few lines of *your own* code. 

#### End to end
Because pre-annotation and active learning need the pipeline to predict structure using input 
format, each preprocessing step must be reverted (sentence splitting, tokenizing, label encoding, ...).

Just as it is done for preprocessing, no magic is involved: just a few explicit instructions that fit your needs exactly and
some pandas-based methods for handling spans and relational structures efficiently.       

#### Caching
Every procedure can be easily cached using and explicit, flexible and performant caching mecanism. Parameter hashing functions have been written to handle numpy, pandas and torch (cuda/cpu) data structures and models seamlessly.

This caching mecanism is useful for checkpointing models, restarting trainings from a given epoch, and instantly preprocessing often used data.

## Examples

Every example takes the form of a standalone notebook to perform simple or complex structuring tasks.
Those are located in the [examples folder](examples).

## Install

This project is still under development and has therefore not been released on the pypi repository.

```bash
pip install git+https://github.com/percevalw/nlstruct.git
```
