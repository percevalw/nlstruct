# Install

Those actions will be replaced in the future by a proper python package installation

Clone
```bash
git clone https://github.com/percevalw/nlstruct.git
```

Add to python path
```
echo 'export PYTHONPATH=$PYTHONPATH:'$pwd'/nlstruct' >> .bashrc
```

Install requirements
```bash
for req in nlstruct/requirements.txt do; pip install $req; done
```

