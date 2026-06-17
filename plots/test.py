from graphviz import Digraph

dot = Digraph(comment="Test")
dot.node("A", "Hello GCA")
dot.render("test_output", format="pdf", cleanup=True)
print("✅ Graphviz 在 Python 中调用完全正常！")