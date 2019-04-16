# -*- coding: utf-8 -*-
from . import helpers
import sys
import os
sys.path.append(os.path.abspath('../sgan'))
# TODO: Importing sgan shouldn't be like this, but SGAN needs to become an installable python package

def get_hmm():
    """Get a thought."""
    return 'hmmm...'


def hmm():
    """Contemplation..."""
    if helpers.get_answer():
        print(get_hmm())
