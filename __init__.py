# see https://docs.python.org/3/library/pkgutil.html
import sys
from pkgutil import extend_path

__path__ = extend_path(sys.path, __name__)
