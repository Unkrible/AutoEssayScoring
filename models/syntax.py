from nltk.parse.stanford import StanfordParser
import os
java_path = "C:/Program Files/Java/jdk1.8.0_231"
os.environ['JAVAHOME'] = java_path
if __name__ == '__main__':
    stanford_parser_dir = 'D:/stanford-parser-full-2018-10-17/'
    eng_model_path = stanford_parser_dir + "stanford-parser-3.9.2-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
    my_path_to_models_jar = stanford_parser_dir + "stanford-parser-3.9.2-models.jar"
    my_path_to_jar = stanford_parser_dir + "stanford-parser.jar"

    parser = StanfordParser(model_path=eng_model_path, path_to_models_jar=my_path_to_models_jar,
                            path_to_jar=my_path_to_jar)

    #parse input is a list, while raw_parse input is a str
    s = parser.parse("you are beautiful when you are young".split())
    #print(s)

    for line in s:
        print("line:",line)
        for t in line:
            print("t:",list(t))
