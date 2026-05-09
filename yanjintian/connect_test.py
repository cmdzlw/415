import webbrowser
from py2neo import Graph, Node, Relationship
from py2neo import Graph

graph = Graph('bolt://localhost:7687', auth=('neo4j', '12345678'))

s1 = Node("Smells", name="Bulk data transfer on slow network", dengji=3)
m1 = Node("Medicine",
          name="Check network connection",
          solution="Another approach is to give the user the choice")

relationship = Relationship(s1, "MEDICINE", m1)
url = "http://localhost:7474/browser/"
webbrowser.open(url)
graph.create(relationship)
