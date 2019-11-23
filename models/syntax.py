from nltk.parse.stanford import StanfordParser


class SyntaxFeature:
    def __init__(self):
        stanford_parser_dir = 'D:/stanford-parser-full-2018-10-17/'
        eng_model_path = stanford_parser_dir + "stanford-parser-3.9.2-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
        my_path_to_models_jar = stanford_parser_dir + "stanford-parser-3.9.2-models.jar"
        my_path_to_jar = stanford_parser_dir + "stanford-parser.jar"

        self.parser = StanfordParser(model_path=eng_model_path, path_to_models_jar=my_path_to_models_jar,
                                     path_to_jar=my_path_to_jar)

    def _printList(self, list1):
        for elements in list1:
            if isinstance(elements, list):
                self._printList(elements)
            else:
                print(elements)

    def _treeDepth(self, Tree):
        if isinstance(Tree, list):
            maxDepth = 0
            for elements in Tree:
                depth = self._treeDepth(elements)
                if depth > maxDepth:
                    maxDepth = depth
            return maxDepth + 1
        else:
            return 0

    def get_tree_depth(self, sents):
        tree = list(self.parser.parse(sents.split()))
        '''
        for line in tree:
            print(line)
        '''
        return self._treeDepth(tree)

    def is_temporal_clauses(self, sents):
        if 'when' in sents or 'When' in sents or 'while' in sents or 'While' in sents or 'as soon as' in sents or 'As soon as' in sents \
                or 'till' in sents or 'Till' in sents or 'until' in sents or 'Until' in sents \
                or ('not' in sents and 'until' in sents) or ('Not' in sents and 'until' in sents) \
                or 'the first time' in sents or 'The first time' in sents or 'the last time' in sents or 'The last time' in sents \
                or 'the moment' in sents or 'The moment' in sents or 'the minute' in sents or 'The minute' in sents \
                or 'the instant' in sents or 'The instant' in sents or 'immediately' in sents or 'Immediately' in sents \
                or 'directly' in sents or 'Directly' in sents or 'instantly' in sents or 'Instantly' in sents \
                or ('hardly' in sents and 'when' in sents) or ('Hardly' in sents and 'when' in sents) \
                or ('no sooner' in sents and 'than' in sents) or ('No sooner' in sents and 'than' in sents) \
                or 'since' in sents or 'Since' in sents or 'by the time' in sents or 'By the time' in sents \
                or 'by the end of' in sents or 'By the end of' in sents:
            return True
        else:
            return False

    def is_causal_clauses(self, sents):
        if 'because' in sents or 'Because' in sents or 'since' in sents or 'Since' in sents or 'now that' in sents or 'Now that' in sents \
                or 'seeing that' in sents or 'Seeing that' in sents or 'considering that' in sents or 'Considering that' in sents \
                or 'given that' in sents or 'Given that' in sents or 'for the reason' in sents or 'For the reason' in sents \
                or 'seeing as' in sents or 'Seeing as' in sents:
            return True
        elif 'so that' in sents or 'So that' in sents or 'such that' in sents or 'Such that' in sents \
                or ('so' in sents and 'that' in sents) or ('So' in sents and 'that' in sents) \
                or ('such' in sents and 'that' in sents) or ('Such' in sents and 'that' in sents):
            return True
        else:
            return False


def treeDepth(Tree):
    if isinstance(Tree, list):
        maxDepth = 0
        for elements in Tree:
            depth = treeDepth(elements)
            if depth > maxDepth:
                maxDepth = depth
        return maxDepth + 1
    else:
        return 0


if __name__ == '__main__':
    '''
    stanford_parser_dir = 'D:/stanford-parser-full-2018-10-17/'
    eng_model_path = stanford_parser_dir + "stanford-parser-3.9.2-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
    my_path_to_models_jar = stanford_parser_dir + "stanford-parser-3.9.2-models.jar"
    my_path_to_jar = stanford_parser_dir + "stanford-parser.jar"

    parser = StanfordParser(model_path=eng_model_path, path_to_models_jar=my_path_to_models_jar,
                            path_to_jar=my_path_to_jar)

    #parse input is a list, while raw_parse input is a str
    #print("you are beautiful when you are young.".split())
    parser_result = parser.parse("you are beautiful when you are young".split())

    tree=list(parser_result)
    # print(s)
    print(treeDepth(tree))
    #printList(list(s))
    print(tree)
    for line in tree:
        print(line)

    '''
    # '''
    sf = SyntaxFeature()
    depth = sf.get_tree_depth("you are beautiful when you are young.")
    print(depth)
    # '''
