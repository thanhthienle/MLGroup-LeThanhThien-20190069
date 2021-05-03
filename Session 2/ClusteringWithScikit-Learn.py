import numpy as np


def load_data(data_path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tdidfs in indices_tfidfs:
            index = int(index_tdidfs.split(':')[0])
            tfidf = float(index_tdidfs.split(':')[1])
            r_d[index] = tfidf
        return np.array(r_d)

    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open('../datasets/20news-bydate/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())

    data = []
    labels = []

    for data_id, d in enumerate(d_lines):
        features = d.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
        data.append(r_d)
        labels.append(label)

    return np.array(data), np.array(labels)


# def clustering_with_KMeans():
#     data, labels = load_data(data_path='../datasets/20news-bydate/data_tf_idf.txt')
#     # use csr_matrix to create a sparse matrix with efficient row slicing
#     from sklearn.cluster import KMeans
#     from scipy.sparse import csr_matrix
#     X = csr_matrix(data)
#     print('=========')
#     kmeans = KMeans(
#         n_clusters=20,
#         init='random',
#         n_init=5,
#         tol=1e-3,
#         random_state=2018
#     ).fit(X)
#     clustered_labels = kmeans.labels_
#
#
# clustering_with_KMeans()


def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float)) / len(expected_y)
    return accuracy


def classifying_with_linear_SVMs():
    train_X, train_Y = load_data(data_path='../datasets/20news-bydate/20news-train-tfidf.txt')
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(
        C=10.0,  # penalty coeff
        tol=0.001,  # tolerance for stopping criteria
        verbose=True    # whether prints out logs or not
    )
    classifier.fit(train_X, train_Y)

    test_X, test_y = load_data(data_path='../datasets/20news-bydate/20news-test-tfidf.txt')
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y=predicted_y, expected_y=test_y)
    print("Accuracy: ", accuracy)


classifying_with_linear_SVMs()


# def classifying_with_kernel_SVMs():
#     train_X, train_Y = load_data(data_path='../datasets/20news-bydate/20news-train-tfidf')
#     from sklearn.svm import SVC
#     classifier = SVC(
#         C=50.0,
#         kernel='rbf',   # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
#         gamma=0.1,
#         tol=0.001,
#         verbose=True
#     )
#     classifier.fit(train_X, train_Y)
#
#     test_X, test_y = load_data(data_path='../datasets/20news-bydate/20news-test-tfidf')
#     predicted_y = classifier.predict(test_X)
#     accuracy = compute_accuracy(predicted_y=predicted_y, expected_y=test_y)
#     print("Accuracy: ", accuracy)


# classifying_with_kernel_SVMs()





