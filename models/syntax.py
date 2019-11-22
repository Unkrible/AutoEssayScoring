'''
import os
from nltk.parse import stanford

#添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
os.environ['STANFORD_PARSER'] = 'D:/stanford-parser-2012-07-09/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'D:/stanford-parser-2012-07-09/stanford-parser-2012-07-09-models.jar'


#为JAVAHOME添加环境变量
java_path = "C:/Program Files (x86)/Java/jdk1.8.0_11/bin/java.exe"
os.environ['JAVAHOME'] = java_path

#句法标注
parser = stanford.StanfordParser(model_path="D:/stanford-parser-2012-07-09/stanford-parser-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
sentences = parser.parse_sents("Hello, My name is Melroy.".split(), "What is your name?".split())
print( sentences)
'''
'''
from nltk.parse import stanford

full_path = u"" #那三个文件的系统全路径目录
parser = stanford.StanfordParser( # 加载解析器，注意：一定要是全路径，从系统根路径开始，不然找不到
path_to_jar=full_path + u"/stanford-parser.jar",
path_to_models_jar=full_path +u"/stanford-parser-3.9.1-models.jar",
model_path=full_path +u'/')

# sentence是分词后的句子，当然它也支持输入未分词的句子，但是一般情况下，我会自己分词
res = list(parser.parse("we are good"))
'''
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

    s = parser.raw_parse("you are beautiful.")
    #print(s)

    for line in s:
        print("line:",line)
        for t in line:
            print("t:",t)
