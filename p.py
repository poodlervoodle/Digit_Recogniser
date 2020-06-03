import numpy as np
import operator


'''Using K Nearest Neighbours intuition'''
def knn(test_row, train, k=5):
    diffs = {}
    i = 0
    for t in train:
        diffs[i] = np.sum(np.power(np.subtract(test_row, t), 2)) #calculating the euclidean distance
        i = i + 1
    return sorted(diffs.items(), key=operator.itemgetter(1))[:k]


# majority vote
def majority(knn, labels):
    a = {}
    for i, d in knn:
        if labels[i] in a.keys():
            a[labels[i]] = a[labels[i]] + 1
        else:
            a[labels[i]] = 1
    return sorted(a.items(), key=operator.itemgetter(1), reverse=True)[0][0]


# worker. crawl through test set and predicts number
def doWork(train, test, labels):
    output_file = open("output.csv", "w")
    i = 0
    size = len(test)
    for test_sample in test:
        i += 1
        start = time.time()
        prediction = majority(knn(test_sample, train, k=100), labels)
        print ("Knn: %f" % (time.time() - start))
        output_file.write(prediction)
        output_file.write("\n")
        print (float(i) / size) * 100
    output_file.close()


# majority vote for a little bit optimized worker
def majority_vote(knn, labels):
    knn = [k[0, 0] for k in knn]
    a = {}
    for i in knn:
        if labels[i] in a.keys():
            a[labels[i]] = a[labels[i]] + 1
        else:
            a[labels[i]] = 1
    return sorted(a.items(), key=operator.itemgetter(1), reverse=True)[0][0]


def doWorkNumpy(train, test, labels):
    k = 20
    train_mat = np.mat(train)
    output_file = open("output-numpy2.csv", "w")
    i = 0
    size = len(test)
    for test_sample in test:
        i += 1
        start = time.time()
        knn = np.argsort(np.sum(np.power(np.subtract(train_mat, test_sample), 2), axis=1), axis=0)[:k]
        s = time.time()
        prediction = majority_vote(knn, labels)
        output_file.write(str(prediction))
        output_file.write("\n")
        print ("Knn: %f, majority %f" % (time.time() - start, time.time() - s))
        print ("Done: %f" % (float(i) / size))
    output_file.close()
    output_file = open("done.txt", "w")
    output_file.write("DONE")
    output_file.close()


if __name__ == '__main__':
    from load_data import read_data
    train, labels = read_data("./train1.csv")
    test, tmpl = read_data("./test1.csv", test=True)
    doWorkNumpy(train, test, labels)