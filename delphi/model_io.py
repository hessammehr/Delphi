import xml.etree.ElementTree as ET

def parse_graph(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    nodes = []
    edges = []

    for mxCell in root.iter('mxCell'):
        if 'edge' in mxCell.attrib:
            edge = {
                'id': mxCell.attrib['id'],
                'from': mxCell.attrib.get('source'),
                'to': mxCell.attrib.get('target')
            }
            edges.append(edge)
        else:
            node = {
                'id': mxCell.attrib['id'],
                'value': mxCell.attrib.get('value')
            }
            nodes.append(node)

    return nodes, edges

if __name__ == "__main__":
    nodes, edges = parse_graph('delphi/example_graph.xml')
    print("Nodes:", nodes)
    print("Edges:", edges)