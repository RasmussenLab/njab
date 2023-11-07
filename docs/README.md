# Docs creation

In order to build the docs you need to 

  1. install sphinx and additional support packages
  2. build the package reference files
  3. run sphinx to create a local html version

The documentation is build using readthedocs automatically.

## Build docs using Sphinx command line tools

Command to be run from `path/to/njab/docs`, i.e. from within the `docs` package folder: 

Options:
  - `--separate` to build separate pages for each (sub-)module

```bash	
# pwd: ./njab/docs
# apidoc
sphinx-apidoc --force --implicit-namespaces --module-first -o reference ../src/njab
# build docs
sphinx-build -n -W --keep-going -b html ./ ./_build/
```