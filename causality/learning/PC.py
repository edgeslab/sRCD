import logging
from ..learning.RCD import RCD
import networkx as nx
from ..model.RelationalDependency import RelationalDependency
from ..model.RelationalDependency import RelationalVariable

logger = logging.getLogger(__name__)

class PC(object):

    def __init__(self, schema, oracle):
        self.schema = schema
        self.oracle = oracle
        self.undirectedSkeleton = None
        self.sepsets = None
        self.rcd = RCD(self.schema, self.oracle, 0)


    def learnModel(self):
        """
        Implements the PC algorithm as outlined in Causation, Predicion and Search, with the orientation rules from Meek 95.
        schema: The relational schema with one entity with the attributes.
        oracle: An oracle that answers d-separation queries.
        returns learnedModel: Partially directed graph that corresponds to the Markov equivalence class of the original model.
        """
        logger.info("Running PC on Schema: {schema}".format(schema=self.schema))
        self.pcPhaseI()
        self.pcPhaseII()


    def pcPhaseI(self):
        """
        Phase I returns an undirected graph and a dictionary that maps node tuples of two to separating sets.
        If the dictionary keys contain (a,b), they also contain (b,a).
        """
        self.rcd.identifyUndirectedDependencies()
        self.sepsets = self.rcd.sepsets
        self.undirectedSkeleton = nx.DiGraph()
        entity = list(self.schema.getEntities())[0]
        entityAttrs = entity.attributes
        self.undirectedSkeleton.add_nodes_from([RelationalVariable([entity.name], attr.name) for attr in entityAttrs])
        self.undirectedSkeleton.add_edges_from([(relDep.relVar1, relDep.relVar2) for relDep in self.rcd.undirectedDependencies])


    def pcPhaseII(self):
        logger.info("Starting PhaseII of PC")
        if not hasattr(self, 'undirectedSkeleton') or self.undirectedSkeleton is None:
            raise Exception("No undirected skeleton found. Try running Phase I first.")
        if not hasattr(self, 'sepsets') or self.sepsets is None:
            raise Exception("No sepsets found. Try running Phase I first.")

        self.rcd.orientDependencies()
        self.partiallyDirectedGraph = nx.DiGraph()
        entity = list(self.schema.getEntities())[0]
        entityAttrs = entity.attributes
        self.partiallyDirectedGraph.add_nodes_from([RelationalVariable([entity.name], attr.name) for attr in entityAttrs])
        self.partiallyDirectedGraph.add_edges_from([(relDep.relVar1, relDep.relVar2)
                                                    for relDep in self.rcd.orientedDependencies])
        logger.info("Result of PhaseII of PC: {edges}".format(edges=self.partiallyDirectedGraph.edges()))


    def setUndirectedSkeleton(self, undirectedSkeleton):
        if not isinstance(undirectedSkeleton, nx.DiGraph):
            raise Exception("Undirected skeleton must be a networkx DiGraph: found {}".format(undirectedSkeleton))

        expectedSkeletonNodes = []
        entities = list(self.schema.getEntities())
        if entities:
            entity = entities[0]
            entityAttrs = entity.attributes
            expectedSkeletonNodes = [RelationalVariable([entity.name], attr.name) for attr in entityAttrs]

        if len(expectedSkeletonNodes) != len(undirectedSkeleton.nodes()) or \
            set(expectedSkeletonNodes) != set(undirectedSkeleton.nodes()):
            raise Exception("Undirected skeleton's nodes must match schema attributes")

        self.undirectedSkeleton = undirectedSkeleton
        self.rcd.undirectedDependencies = [RelationalDependency(relVar1, relVar2) for relVar1, relVar2 in
                                           self.undirectedSkeleton.edges()]
        self.rcd.constructAggsFromDependencies(self.rcd.undirectedDependencies)


    def setSepsets(self, sepsets):
        if not isinstance(sepsets, dict):
            raise Exception("Sepsets must be a dictionary: found {}".format(sepsets))

        self.rcd.setSepsets(sepsets)
        self.sepsets = self.rcd.sepsets