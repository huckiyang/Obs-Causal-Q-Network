#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
try:
    import warnings
    from IPython import get_ipython
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict
    import json
    import numpy as np
    import pandas as pd
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO
    warnings.filterwarnings('ignore', category=UserWarning,
                            module='matplotlib')
    # Display inline matplotlib plots with IPython
    get_ipython().run_line_magic('matplotlib', 'inline')
    # aesthetics
    sns.set_palette('deep', desat=.6)
    sns.set_context(rc={'figure.figsize': (8, 4)})
    sns.set_style('whitegrid')
    sns.set_palette(sns.color_palette('Set2', 10))
    # loading style sheet
    get_ipython().run_cell('from IPython.core.display import HTML')
    get_ipython().run_cell('HTML(open("style/ipython_style.css").read())')
except:
    pass
###########################################

'''
Begin help functions
'''


'''
End help functions
'''
