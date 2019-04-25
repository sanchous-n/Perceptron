import numpy


class Perceptron:

    def __init__(self, learning_rate=0.01, epoch=100):
        self.learning_rate = learning_rate
        self.weights = ([], 0)
        self.epoch = epoch

    def predict(self, inputs):
        summation = numpy.dot(inputs, self.weights[0])
        if summation > self.weights[1]:
            return 1
        return 0

    def train(self, training_inputs):
        self.weights = training_inputs[0]
        for _a in range(self.epoch):
            for tr in training_inputs[1:]:
                prediction = self.predict(tr[0])
                wList = [w + (x * self.learning_rate * (tr[1] - prediction)) for w, x in zip(self.weights[0], tr[0])]
                wLab = self.weights[1] - self.learning_rate * (tr[1] - prediction)
                self.weights = (wList, wLab)

    def testForAccuracy(self, ls):
        correct = 0
        fn = 0
        for i in ls:
            if i[1] == self.predict(i[0]):
                correct += 1
            if i[1] == 1 and self.predict(i[0]) == 0:
                fn += 1
        per = 100 * correct / len(ls)
        rec = correct / (fn+correct)
        print(f"Accuracy for given set is {per}%")
        print(f"Recall for given set is {rec}")
