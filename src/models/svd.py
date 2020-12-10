import pandas as pd
from surprise import SVD


def train_svd(trainset, random_state):
    algorithm = SVD(random_state=random_state)
    algorithm.fit(trainset)
    return algorithm


def recommend_svd(algorithm, ui, ratings_f, title_to_id):
    agg = ratings_f.groupby('userId')['movieId'].agg(list)

    if ui in ratings_f.userId.unique():
        ui_list = agg[ui]  # Из таблицы ratings_f создайте list фильмов, которые оценил конкретный пользователь
        d = {k: v for k, v in title_to_id.items() if
             v not in ui_list}  # Создайте инвертированный словарь title_to_id, но только для тех фильмов, которые ui не видел. То есть исключите
        # из списка фильмов множество ui_list. Инвертированный - то есть ключи стали значениями, а значения - ключами

        # С помощью нашей обученной модели, проставим предсказанные рейтинги фильмам, которые пользователь еще не видел.
        predictedL = []
        for i, j in d.items():
            predicted = algorithm.predict(ui, j)
            predictedL.append((i, predicted[3]))
        pdf = pd.DataFrame(predictedL, columns=['movies',
                                                'ratings'])  # Создайте датафрейм из массива predictedL с колонками ['movies', 'ratings']

        pdf = pdf.sort_values(['ratings'],
                              ascending=False)  # Отсортируйте таблицу по колонке ratings, от большего - к меньшему

        pdf.set_index('movies', inplace=True)
        return pdf.head(10)
    else:
        print("Пользователь не найден в списке!")
        return None
