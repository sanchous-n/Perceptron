from nai2 import Perceptron


def getDataRowList(path):
    with open(path) as f:
        tList = f.readlines()

    return [x.rstrip() for x in tList]


def getDataDoneList(tList):
    trainDataList = []
    for i in tList:
        tData = i.split(",")
        trainDataList.append(([float(i) for i in tData[:-1]], int(tData[-1])))

    return trainDataList


def getFlowerDoneList(tList):
    trainDataList = []
    dict = {"Iris-virginica": 0, "Iris-versicolor": 1}
    for i in tList:
        tData = i.split(",")
        lb = dict[tData[-1]]
        trainDataList.append(([float(i) for i in tData[:-1]], int(lb)))

    return trainDataList


def numberTest(al, ep):
    perc = Perceptron.Perceptron(al, ep)
    print("\nTesting set of numbers:")
    perc.train(getDataDoneList(getDataRowList(r'..\Data\nums\train.txt')))
    perc.testForAccuracy(getDataDoneList(getDataRowList(r"..\Data\nums\test.txt")))
    if "y" == input("\nInput data manually? y/n  "):
        test = [float(x) for x in input("Enter data: ").rstrip().split(",")]
        print(f"Prediction is {perc.predict(test)}")


def flowerTest(al, ep):
    perc = Perceptron.Perceptron(al, ep)
    dictF = {0: "Iris-virginica", 1: "Iris-versicolor"}

    print("\nTesting set of flowers:")
    perc.train(getFlowerDoneList(getDataRowList(r'..\Data\iris_per\training.txt')))
    perc.testForAccuracy(getFlowerDoneList(getDataRowList(r'..\Data\iris_per\test.txt')))
    if "y" == input("\nInput data manually? y/n  "):
        test = [float(x) for x in input("Enter data: ").rstrip().split(",")]
        print(f"Prediction is {dictF[perc.predict(test)]}")


def main():
    ep = int(input("Input epoch: "))
    al = float(input("Input learning rate: "))
    numberTest(al, ep)
    print("\n--------------------------------------------")
    flowerTest(al, ep)


if __name__ == "__main__":
    main()
