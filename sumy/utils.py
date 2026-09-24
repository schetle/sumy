from os.path import dirname, abspath, join, exists


def cached_property(getter):
    """
    Decorator that converts a method into memoized property.
    The decorator works as expected only for classes with
    attribute '__dict__' and immutable properties.
    For __slots__ classes, we use a simpler approach.
    """
    attr_name = "_cached_property_" + getter.__name__
    
    @property
    def decorator(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, getter(self))
        return getattr(self, attr_name)
    
    return decorator


def expand_resource_path(path):
    directory = dirname(__file__)
    directory = abspath(directory)
    return join(directory, "data", path)


def get_stop_words(language):
    path = expand_resource_path("stopwords/%s.txt" % language)
    if not exists(path):
        raise LookupError("Stop-words are not available for language %s." % language)
    return read_stop_words(path)


def read_stop_words(filename):
    with open(filename, "rb") as open_file:
        return frozenset(w.rstrip().decode("utf8") for w in open_file.readlines())


class ItemsCount:
    def __init__(self, value):
        self._value = value

    def __call__(self, sequence):
        if isinstance(self._value, str):
            if self._value.endswith("%"):
                total_count = len(sequence)
                percentage = int(self._value[:-1])
                # at least one sentence should be choosen
                count = max(1, total_count*percentage // 100)
                return sequence[:count]
            else:
                return sequence[:int(self._value)]
        elif isinstance(self._value, (int, float)):
            return sequence[:int(self._value)]
        else:
            ValueError("Unsuported value of items count '%s'." % self._value)

    def __repr__(self):
        return "<ItemsCount: %r>" % self._value
