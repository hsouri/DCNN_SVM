import mnist_reader
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from Classifiers import SVM


def main():
    file = 'out.txt'
    train_data, train_labels = mnist_reader.load_mnist('data/fashion', kind='train')
    train_data = 5 * (train_data / 255 - 0.5) / 0.5
    test_data, test_labels = mnist_reader.load_mnist('data/fashion', kind='t10k')
    test_data = 5 * (test_data / 255 - 0.5) / 0.5

    clf = SVM(train_data, train_labels, test_data, test_labels, file)
    clf.train('linear').test()
    clf.train('polynomial').test()
    clf.train('RBF').test()

    pca = PCA(n_components=100)
    pca.fit(train_data)
    pca_train_data = pca.transform(train_data)
    pca_test_data = pca.transform(test_data)
    clf = SVM(pca_train_data, train_labels, pca_test_data, test_labels, file)
    clf.train('linear').test()
    clf.train('polynomial').test()
    clf.train('RBF').test()

    lda = LDA(n_components=9)
    lda.fit(train_data, train_labels)
    lda_train_data = lda.transform(train_data)
    lda_test_data = lda.transform(test_data)
    clf = SVM(lda_train_data, train_labels, lda_test_data, test_labels, file)
    clf.train('linear').test()
    clf.train('polynomial').test()
    clf.train('RBF').test()


if __name__ == '__main__':
    main()