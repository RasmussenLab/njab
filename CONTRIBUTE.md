# Development tools

## Using tox with anaconda

[tox](https://tox.wiki/en/latest/installation.html) is recommended to use `pipx`.
tox does not seem to work out of the box
with conda based python installation.

Install pipx into your base environment:

```bash
# to base
conda install -c defaults -c conda-forge pipx
```

```
pipx install tox
pipx ensurepath # add to path
```

Start a new shell. Now tox should work from your base environment.

