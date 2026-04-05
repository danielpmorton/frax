"""Assorted utility functions"""


def tuplify(arr):
    """Recursively convert a nested structure (for instance, lists, tuples, arrays) to a tuple.
    If arr is not either of these, it returns the original value.

    For instance,
    ```
    tuplify([1, 2, [3, 4]]) == (1, 2, (3, 4))
    tuplify(np.array([[1, 2], [3, 4]])) == ((1, 2), (3, 4))
    tuplify(1) == 1
    tuplify("hello") == "hello"
    ```

    Args:
        arr (Any): A (possibly) nested structure of lists, tuples, arrays, etc.

    Returns:
        Any: All lists/tuples/arrays converted to tuples; other values unchanged
    """
    if isinstance(arr, (list, tuple)):
        return tuple(tuplify(a) for a in arr)
    elif hasattr(arr, "tolist") and callable(arr.tolist):
        # This handles numpy and jax arrays
        return tuplify(arr.tolist())
    else:
        return arr
