from models.feature_common import *
from models.feature_engineering import *
import os,jpype
import pandas as pd
import numpy as np
import progressbar


class StringKernelFeatureEngineering(FeatureEngineer):
    def __init__(self):
        FeatureEngineer.__init__(self)
        self.jarpath = os.path.join(os.path.abspath('../resources/StringKernelsPackage/code'), 'test.jar')
        if not jpype.isJVMStarted():
            jpype.startJVM(jpype.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % self.jarpath)
        BlendedIntersectionStringKernel = jpype.JClass('BlendedIntersectionStringKernel')
        self.bisk = BlendedIntersectionStringKernel(1, 15)

    def __del__(self):
        jpype.shutdownJVM()

    @timeit
    def fit(self, data):
        pass

    @timeit
    def transform(self, data):
        data = data[:]
        data = pd.DataFrame({'essay_id': data.index, 'essay': data.values})
        # data['essay'] = data.apply(lambda x: x['essay'].strip(), axis=1)

        docs_num = len(data)
        matrix = np.zeros((docs_num, docs_num), dtype=np.long)
        for i in range(docs_num):
            for j in range(i, docs_num):
                score = self.bisk.computeKernel(data.at[i, 'essay'], data.at[j, 'essay'])
                matrix[i][j], matrix[j][i] = score, score

        return data['essay_id'], matrix


    @timeit
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def read_labels(label_path, set_index, columns=('essay_id', 'essay_set', 'domain1_score')):
    df = pd.read_csv(label_path, sep='\t')
    for each in df.columns.values.tolist():
        if each not in columns:
            df.pop(each)
    df = df[df['essay_set'] == set_index]
    df.pop('essay_set')
    return df


if __name__=='__main__':
    # jarpath = os.path.join(os.path.abspath('../resources/StringKernelsPackage/code'), 'test.jar')
    # print(jarpath)
    # jpype.startJVM(jpype.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % jarpath)
    # BlendedIntersectionStringKernel = jpype.JClass('BlendedIntersectionStringKernel')
    # bisk = BlendedIntersectionStringKernel(1, 15)
    # print(bisk.computeKernel("This is yanfan.", "This is yanfan."))
    # print('Done, over')
    fe = StringKernelFeatureEngineering()
    for set_id in range(1, 9):
        print("Now for set %d" % set_id)
        dataset = AutoEssayScoringDataset("../resources/essay_data", essay_set_id=set_id)
        train, _ = dataset.train
        valid, _ = dataset.valid
        test = dataset.test.data

        # train_from, train_to, valid_from, valid_to, test_from = \
        #     0, len(train), len(train),  len(train)+len(valid), len(train)+len(valid)
        train_from, train_to, valid_from, valid_to, test_from = 0, 3, 3, 6, 6
        dataset = pd.concat([train, valid, test])

        essay_id, matrix = fe.fit_transform(dataset)

        train = pd.DataFrame(matrix[train_from:train_to])
        train['essay_id'] = essay_id[train_from:train_to]
        valid = pd.DataFrame(matrix[valid_from:valid_to])
        valid['essay_id'] = essay_id[valid_from:valid_to]
        test = pd.DataFrame(matrix[test_from:])
        test['essay_id'] = essay_id[test_from:]

        train.to_csv('../resources/hisk/TrainSet' + str(set_id) + '.csv', index=False)
        train_label = read_labels('../resources/essay_data/train.tsv', set_id)
        train_label.to_csv('../resources/hisk/TrainLabel' + str(set_id) + '.csv', index=False)

        valid.to_csv('../resources/hisk/ValidSet' + str(set_id) + '.csv', index=False)
        valid_label = read_labels('../resources/essay_data/dev.tsv', set_id)
        valid_label.to_csv('../resources/hisk/ValidLabel' + str(set_id) + '.csv', index=False)

        test.to_csv('../resources/hisk/TestSet' + str(set_id) + '.csv', index=False)
        test_label = read_labels('../resources/essay_data/test.tsv', set_id)
        test_label.to_csv('../resources/hisk/TestLabel' + str(set_id) + '.csv', index=False)

    print("Done~")
