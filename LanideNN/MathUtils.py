import collections


def flatten(matrix):
    """Flatten any nested lists

    Args:
      matrix: input matrix.

    Returns:
      generator over all its items
    """
    for el in matrix:
        if isinstance(el, collections.Iterable) and not isinstance(el, list):
            for sub in flatten(el):
                yield sub
        else:
            yield el
