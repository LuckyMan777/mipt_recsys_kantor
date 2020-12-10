from surprise import accuracy


def evaluate_svd(algorithm, testset):
    predictions = algorithm.test(testset)

    return accuracy.rmse(predictions)
