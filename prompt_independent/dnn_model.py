from models.classifier import Classifier


# TODO: 从相应的csv文件中提取特征
class DNNModel(Classifier):
    def __init__(self, **kwargs):
        super(DNNModel, self).__init__()

    def fit(self, dataset, *args, **kwargs):
        pass

    def predict(self, data, *args, **kwargs):
        pass
