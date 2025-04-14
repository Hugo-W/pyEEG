def hook(config):
    import numpy
    for ext in config.get('tool.setuptools.ext_modules', []):
        ext.setdefault('include_dirs', []).append(numpy.get_include())