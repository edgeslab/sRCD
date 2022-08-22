from ..model.Schema import SchemaItem

class RelationalVariable(object):

    def __init__(self, relPath, attrName):
        """
        relPath: an alternating sequence of entity and relationship names
        attrName
        NB: When used to represent attributes to generate, the first item is always the schema item
            on which the attribute exists
        """
        self.path = relPath
        self.attrName = attrName


    def __key(self):
        return tuple(self.path), self.attrName


    def __eq__(self, other):
        return isinstance(other, RelationalVariable) and self.__key() == other.__key()


    def __hash__(self):
        return hash(self.__key())


    def __lt__(self, other):
        if not isinstance(other, RelationalVariable) and not isinstance(other, RelationalVariableIntersection):
            raise TypeError("unorderable types: RelationalVariable() < {}()".format(type(other)))
        if isinstance(other, RelationalVariable):
            return self.__key() < other.__key()
        if isinstance(other, RelationalVariableIntersection):
            return self.__key() < other._RelationalVariableIntersection__key()


    def __repr__(self):
        return "{}.{}".format(str(self.path).replace("'", ""), self.attrName)


    def getBaseItemName(self):
        return self.path[0]


    def getTerminalItemName(self):
        return self.path[-1]


    def isExistence(self):
        return self.attrName == SchemaItem.EXISTS_ATTR_NAME


    def intersects(self, other):
        return self.getBaseItemName() == other.getBaseItemName() \
            and self.getTerminalItemName() == other.getTerminalItemName() \
            and self.attrName == other.attrName \
            and any([item1 != item2 for item1, item2 in zip(self.path, other.path)])


class RelationalDependency(object):

    TAIL_MARK_EMPTY = ''
    TAIL_MARK_CIRCLE = 'o'
    TAIL_MARK_LEFT_ARROW = '<'
    TAIL_MARK_RIGHT_ARROW = '>'

    def __init__(self, relVar1, relVar2, tailMarkFrom=TAIL_MARK_EMPTY, tailMarkTo=TAIL_MARK_RIGHT_ARROW):
        if not isinstance(relVar1, RelationalVariable) or not isinstance(relVar2, RelationalVariable):
            raise Exception("RelationalDependency expects two RelationalVariable objects")

        self.relVar1 = relVar1
        self.relVar2 = relVar2
        self.tailMarkFrom = tailMarkFrom
        self.tailMarkTo = tailMarkTo


    def __key(self):
        return self.relVar1._RelationalVariable__key(), self.relVar2._RelationalVariable__key()


    def __eq__(self, other):
        # if not isinstance(other, RelationalDependency):
        #     return False
        # if self.tailMarkFrom == self.tailMarkTo or \
        #     other.tailMarkFrom == other.tailMarkTo or \
        #     (self.tailMarkFrom == RelationalDependency.TAIL_MARK_LEFT_ARROW and self.tailMarkTo == RelationalDependency.TAIL_MARK_RIGHT_ARROW) or \
        #     (other.tailMarkFrom == RelationalDependency.TAIL_MARK_LEFT_ARROW and other.tailMarkTo == RelationalDependency.TAIL_MARK_RIGHT_ARROW):     # - / o-o / <->
        #     #[B].Y1 <-> [B].Y2 == [B].Y2 <-> [B].Y1
        #     return self.__key() == other.__key() or self.__key()[::-1] == other.__key()
        # return self.__key() == other.__key()
        return isinstance(other, RelationalDependency) and self.__key() == other.__key()


    def __hash__(self):
        if self.isFeedbackLoop():
            return hash(min(self.__key(), self.reverse().__key()))
        return hash(self.__key())


    def __lt__(self, other):
        if not isinstance(other, RelationalDependency):
            raise TypeError("unorderable types: RelationalDependency() < {}()".format(type(other)))
        return self.__key() < other.__key()


    def __repr__(self):
        return "{} {}-{} {}".format(self.relVar1, self.tailMarkFrom, self.tailMarkTo, self.relVar2)


    def reverse(self):
        newRelVar1Path = self.relVar1.path[:]
        newRelVar1Path.reverse()
        newRelVar2Path = [self.relVar1.getTerminalItemName()]
        newRelVar1AttrName = self.relVar2.attrName
        newRelVar2AttrName = self.relVar1.attrName
        return RelationalDependency(RelationalVariable(newRelVar1Path, newRelVar1AttrName),
                                    RelationalVariable(newRelVar2Path, newRelVar2AttrName))


    def mirror(self):
        newRelVar1Path = self.relVar1.path[:]
        newRelVar1Path.reverse()
        newRelVar2Path = [self.relVar1.getTerminalItemName()]
        newRelVar1AttrName = self.relVar2.attrName
        newRelVar2AttrName = self.relVar1.attrName
        newTailMarkFrom = self.tailMarkTo
        newTailMarkTo = self.tailMarkFrom

        if newTailMarkFrom == RelationalDependency.TAIL_MARK_RIGHT_ARROW:
            newTailMarkFrom = RelationalDependency.TAIL_MARK_LEFT_ARROW

        if newTailMarkTo == RelationalDependency.TAIL_MARK_LEFT_ARROW:
            newTailMarkTo = RelationalDependency.TAIL_MARK_RIGHT_ARROW

        return RelationalDependency(RelationalVariable(newRelVar1Path, newRelVar1AttrName),
                                    RelationalVariable(newRelVar2Path, newRelVar2AttrName),
                                    newTailMarkFrom, newTailMarkTo)


    def isFeedbackLoop(self):
        return self.tailMarkFrom == RelationalDependency.TAIL_MARK_LEFT_ARROW and \
            self.tailMarkTo == RelationalDependency.TAIL_MARK_RIGHT_ARROW


class RelationalVariableIntersection(object):

    def __init__(self, relVar1, relVar2):
        if not isinstance(relVar1, RelationalVariable) or not isinstance(relVar2, RelationalVariable):
            raise Exception("RelationalVariableIntersection expects two RelationalVariable objects")

        self.relVar1 = relVar1
        self.relVar2 = relVar2


    def __key(self):
        return self.relVar1._RelationalVariable__key() + self.relVar2._RelationalVariable__key() \
            if self.relVar1 < self.relVar2 \
            else self.relVar2._RelationalVariable__key() + self.relVar1._RelationalVariable__key()


    def __eq__(self, other):
        return isinstance(other, RelationalVariableIntersection) and self.__key() == other.__key()


    def __hash__(self):
        return hash(self.__key())


    def __lt__(self, other):
        if not isinstance(other, RelationalVariableIntersection) and not isinstance(other, RelationalVariable):
            raise TypeError("unorderable types: RelationalVariableIntersection() < {}()".format(type(other)))
        if isinstance(other, RelationalVariableIntersection):
            return self.__key() < other.__key()
        if isinstance(other, RelationalVariable):
            return self.__key() < other._RelationalVariable__key()


    def __repr__(self):
        return "<{} {!r}, {!r}>".format(self.__class__.__name__, self.relVar1, self.relVar2)
