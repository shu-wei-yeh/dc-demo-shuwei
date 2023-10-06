from bokeh.io import save as save_
from bokeh.resources import CDN


def save(layouts, fname, title):
    save_(layouts, fname, title=title, resources=CDN)
