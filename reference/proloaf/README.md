# Documentation goes here

This folder is where the .html files for the python reference documentation produced by pdoc (or via any other methods) should be placed, so that Hugo knows to include them as static content. Links to this content on several pages of the website assume the existence of this folder.

When using pdoc specifically, this directory should contain the "topmost" index.html of the module as well as subdirectories for the 'source' and 'proloaf' submodules.

This can be achieved by navigating to the "proloaf" repository and then running:

```bash
pdoc --html MYPATH --output-dir docs\static\reference --force
```

where MYPATH is the path to the "proloaf" repository.