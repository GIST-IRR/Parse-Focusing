from constituent_treelib import ConstituentTree, Language
from nltk import Tree


def main():
    nlp = ConstituentTree.create_pipeline(
        Language.English, ConstituentTree.SpacyModelSize.Medium
    )

    t = Tree.fromstring("test")

    for tp in t.treepositions():
        if not isinstance(t[tp], str):
            t[tp].set_label(t[tp].label()[1:-1])
        tree = ConstituentTree(t, nlp)
    tree.export_tree("test_tree.tex")


if __name__ == "__main__":
    main()
