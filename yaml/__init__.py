
from .error import *

from .tokens import *
from .events import *
from .nodes import *

from .loader import *

__version__ = '3.13'
#try:
#    from .cyaml import *
#    __with_libyaml__ = True
#except ImportError:
__with_libyaml__ = False

#import io

def scan(stream, xLoader=Loader):
    """
    Scan a YAML stream and produce scanning tokens.
    """
    loader = xLoader(stream)
    try:
        while loader.check_token():
            yield loader.get_token()
    finally:
        loader.dispose()

def parse(stream, xLoader=Loader):
    """
    Parse a YAML stream and produce parsing events.
    """
    loader = xLoader(stream)
    try:
        while loader.check_event():
            yield loader.get_event()
    finally:
        loader.dispose()

def compose(stream, xLoader=Loader):
    """
    Parse the first YAML document in a stream
    and produce the corresponding representation tree.
    """
    loader = xLoader(stream)
    try:
        return loader.get_single_node()
    finally:
        loader.dispose()

def compose_all(stream, xLoader=Loader):
    """
    Parse all YAML documents in a stream
    and produce corresponding representation trees.
    """
    loader = xLoader(stream)
    try:
        while loader.check_node():
            yield loader.get_node()
    finally:
        loader.dispose()

def load(stream, xLoader=Loader):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.
    """
    loader = xLoader(stream)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()

def load_all(stream, xLoader=Loader):
    """
    Parse all YAML documents in a stream
    and produce corresponding Python objects.
    """
    loader = xLoader(stream)
    try:
        while loader.check_data():
            yield loader.get_data()
    finally:
        loader.dispose()

def safe_load(stream):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.
    Resolve only basic YAML tags.
    """
    return load(stream, SafeLoader)

def safe_load_all(stream):
    """
    Parse all YAML documents in a stream
    and produce corresponding Python objects.
    Resolve only basic YAML tags.
    """
    return load_all(stream, SafeLoader)


