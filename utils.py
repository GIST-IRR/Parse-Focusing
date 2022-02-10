from nltk import Tree

def depth_from_span(span):
    tree = span_to_tree(span)
    return tree.height() + 1

def span_to_tree(span):
    return Tree.fromlist(span_to_list(span))

def span_to_list(span):
    root = span[0]
    label = root[2] if len(root) >= 3 else 'N'
    if len(span) == 1:
        return [label, ['T']]
    left_child = span[1]
    if len(span) == 2:
        return [label, span_to_list([left_child])]
    others = span[2:]

    sibling_index = []
    end = left_child[1]
    for n in others:
        if n[0] >= end:
            idx = others.index(n)
            sibling_index.append(idx)
            end = n[1]

    children = []
    if len(sibling_index) > 0:
        left_child = [left_child] + others[:sibling_index[0]]
    else:
        left_child = [left_child] + others
    children.append(span_to_list(left_child))

    for i in range(len(sibling_index)):
        if i == len(sibling_index) - 1:
            child = others[sibling_index[i]:]
            children.append(span_to_list(child))
        else:
            child = others[sibling_index[i]:sibling_index[i+1]]
            children.append(span_to_list(child))
    return [label] + children