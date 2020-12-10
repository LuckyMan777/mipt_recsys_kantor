def get_recommendations(title, metadata, indices, cosine_sim):
    idx = indices[title]  # Получите индекс по названию

    sim_scores = list(enumerate(cosine_sim[idx]))  # Список оценок сходства для фильма по его индексу из матрицы оценок

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Отсортируйте массив по скорам (второй элемент)

    sim_scores = [x for x in sim_scores[:11] if x[0] != idx]  # Возьмите 10 первых элементов (кроме самого фильма)

    movie_indices = [x[0] for x in sim_scores]  # Получите массив индексов этих 10 элементов

    # Верните названия топ10 похожих фильмов
    return metadata['title'].iloc[movie_indices]
