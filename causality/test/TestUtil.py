import inspect
import os


def assertRaisesMessage(testCase, excClass, message, callableObj=None, *args, **kwargs):
    try:
        callableObj(*args, **kwargs)
        testCase.fail()
    except excClass as exc:
        testCase.assertEqual(message, None if len(exc.args) == 0 else exc.args[0])


def assertUnorderedListEqual(testCase, list1, list2):
    # sort doesn't work for mixed types. Count occurrences of each element in list instead
    if type(list1) != list or type(list2) != list:
        testCase.fail("list1 or list2 is not a list.")

    if all([type(el) == list for el in list1]) and all([type(el) == list for el in list2]):
        list1Copy = [tuple(el) for el in list1]
        list2Copy = [tuple(el) for el in list2]
    else:
        list1Copy, list2Copy = list1, list2

    bucketDict1 = {}
    bucketDict2 = {}
    for item in list1Copy:
        bucketDict1.setdefault(item, 0)
        bucketDict1[item] += 1
    for item in list2Copy:
        bucketDict2.setdefault(item, 0)
        bucketDict2[item] += 1
    # following works around Issue #10017, issue #14998, which are fixed in 3.3.0: Fix TypeError using pprint on dictionaries with user-defined types as keys or other unorderable keys.
    try:
        testCase.assertEqual(bucketDict1, bucketDict2)
    except TypeError as te:     # TypeError: unorderable types: int() < str()
        testCase.fail("dictionaries not equal, but can't pprint due to python 3.2 error")


def get_test_dir_path():
    """
    Returns the directory that contains this file. Usefil for finding test support files.
    """
    return os.path.abspath(os.path.dirname(inspect.getsourcefile(assertRaisesMessage)))
