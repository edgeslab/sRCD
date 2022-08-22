import re
from ..model.RelationalDependency import RelationalVariable
from ..model.RelationalDependency import RelationalDependency

def parseRelVar(relVarStr):
    """
    Returns a RelationalVariable corresponding to relVarStr, a string representation of a relational variable.
    Examples: '[A].X', '[A, AB].XY'
    If relDep is already a RelationalVariable object, just return it.
    """
    if not isinstance(relVarStr, str) and not isinstance(relVarStr, RelationalVariable):
        raise Exception("relVarStr {!r} is not a string or RelationalVariable Object".format(str(relVarStr)))
    if isinstance(relVarStr, RelationalVariable):
        return relVarStr
    if relVarStr.count('.') != 1:
        raise Exception("relVarStr {!r} did not have exactly one dot".format(relVarStr))
    if relVarStr.count('[') != 1:
        raise Exception("relVarStr {!r} did not have exactly one left square bracket".format(relVarStr))
    if relVarStr.count(']') != 1:
        raise Exception("relVarStr {!r} did not have exactly one right square bracket".format(relVarStr))
    pathStr, attrName = relVarStr.split('.')
    if pathStr[0] != '[' or pathStr[-1] != ']':
        raise Exception("pathStr {!r} did not start and end with square brackets".format(pathStr))

    pathStr = pathStr[1:-1]
    itemNames = pathStr.split(',')
    return RelationalVariable([itemName.strip() for itemName in itemNames], attrName.strip())


def parseRelDep(relDepStr):
    """
    Returns a RelationalDependency instance corresponding to relDepStr, a string representation of a relational dependency.
    Examples: '[A].X -> [A].Y', '[A, AB, B].Y -> [A, AB].XY'
    If relDepStr is already a RelationalDependency object, just return it.
    """
    if not isinstance(relDepStr, str) and not isinstance(relDepStr, RelationalDependency):
        raise Exception("relDepStr is not a string or RelationalDependency object")
    if isinstance(relDepStr, RelationalDependency):
        return relDepStr
    if relDepStr.count('->') != 1:
        raise Exception("relDepStr {!r} did not have exactly one '->' arrow".format(relDepStr))
    relVar1, relVar2 = [parseRelVar(relVarStr.strip()) for relVarStr in relDepStr.split('->')]
    return RelationalDependency(relVar1, relVar2)
