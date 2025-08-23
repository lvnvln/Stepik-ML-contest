# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 14:00:41 2025

@author: Polina
"""

#!/usr/bin/env python
# coding: utf-8

"""
Прогнозирование успешного завершения онлайн-курса "Анализ данных в R"
По данным о первых двух днях активности студентов предсказывается,
наберут ли они более 40 правильных решений за весь курс.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
def load_data():
    """Загрузка тренировочных и тестовых данных"""
    events_data_train = pd.read_csv('data/event_data_train.csv')
    submissions_data_train = pd.read_csv('data/submissions_data_train.csv')
    events_data_test = pd.read_csv('data/events_data_test.csv')
    submissions_data_test = pd.read_csv('data/submission_data_test.csv')
    return events_data_train, submissions_data_train, events_data_test, submissions_data_test

def data_preparation(events_data, submissions_data):
    """
    Создание базового датасета с действиями пользователей

    Parameters:
    events_data: данные о событиях
    submissions_data: данные о решениях

    Returns:
    users_data: объединенный датасет с фичами
    """
    # Создаем базовый датасет со всеми действиями юзеров
    users_events_data = events_data.pivot_table(
        index='user_id',
        columns='action',
        values='step_id',
        aggfunc='count',
        fill_value=0
    ).reset_index()

    # Датасет с количеством правильных и неправильных попыток
    users_scores = submissions_data.pivot_table(
        index='user_id',
        columns='submission_status',
        values='step_id',
        aggfunc='count',
        fill_value=0
    ).reset_index()

    # Соединяем в один датасет
    users_data = users_scores.merge(users_events_data, on='user_id', how='outer')
    users_data = users_data.fillna(0)

    return users_data

def create_time_features(events_data):
    """
    Создание временных фич

    Parameters:
    events_data: данные о событиях

    Returns:
    users_time_feature: датасет с временными характеристиками
    """
    events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
    events_data['day'] = events_data.date.dt.date

    # Таблица с первым и последним действием юзера и количеством уникальных дней
    users_time_feature = events_data.groupby('user_id').agg({
        'timestamp': ['min', 'max'],
        'day': 'nunique'
    }).droplevel(level=0, axis=1).rename(columns={
        'nunique': 'days'
    }).reset_index()

    return users_time_feature

def time_filter(data, days=2):
    """
    Фильтрация данных по временному окну (первые N дней)

    Parameters:
    data: входные данные
    days: количество дней для анализа

    Returns:
    data_with_time_filter: отфильтрованные данные
    """
    user_min_time = data.groupby('user_id').agg({
        'timestamp': 'min'
    }).rename({'timestamp': 'min_timestamp'}, axis=1).reset_index()

    data_with_time_filter = data.merge(user_min_time, on='user_id', how='outer')

    # Отбираем записи не позднее N дней с начала учебы
    time_threshold = days * 24 * 60 * 60  # перевели в секунды
    data_with_time_filter = data_with_time_filter.query(
        "timestamp <= min_timestamp + @time_threshold"
    )
    data_with_time_filter = data_with_time_filter.drop('min_timestamp', axis=1)

    return data_with_time_filter

def passed_course_feature(submission_data, threshold=40):
    """
    Создание целевой переменной - прохождение курса

    Parameters:
    submission_data: данные о решениях
    threshold: порог правильных решений

    Returns:
    users_count_correct: датасет с целевой переменной
    """
    users_count_correct = submission_data[submission_data.submission_status == 'correct'] \
        .groupby('user_id') \
        .agg({'step_id': 'count'}) \
        .reset_index() \
        .rename(columns={'step_id': 'corrects'})

    users_count_correct['passed_course'] = (users_count_correct.corrects >= threshold).astype('int')
    users_count_correct = users_count_correct.drop('corrects', axis=1)

    return users_count_correct

def steps_tried_feature(submissions_data):
    """
    Количество уникальных шагов, которые попробовал пользователь

    Parameters:
    submissions_data: данные о решениях

    Returns:
    steps_tried: датасет с количеством попыток
    """
    steps_tried = submissions_data.groupby('user_id').step_id.nunique() \
        .to_frame().reset_index().rename(columns={'step_id': 'steps_tried'})

    return steps_tried

def correct_ratio_feature(data):
    """
    Расчет доли правильных ответов

    Parameters:
    data: входные данные

    Returns:
    data: данные с добавленной фичей correct_ratio
    """
    data['correct_ratio'] = (data.correct / (data.correct + data.wrong)).fillna(0)
    return data

def create_dataframe(events_data, submission_data):
    """
    Формирование финального датасета для обучения

    Parameters:
    events_data: данные о событиях
    submission_data: данные о решениях

    Returns:
    X: матрица признаков
    y: целевая переменная
    """
    # Фильтрация данных за два дня
    events_time_filter = time_filter(events_data)
    submissions_time_filter = time_filter(submission_data)

    # Создание таблиц с фичами
    users_data = data_preparation(events_time_filter, submissions_time_filter)
    users_passed_course = passed_course_feature(submission_data)
    users_time_feature = create_time_features(events_time_filter)
    users_steps_tried = steps_tried_feature(submissions_time_filter)
    users_data = correct_ratio_feature(users_data)

    # Объединение всех фич
    first_merge = users_data.merge(users_steps_tried, how='outer').fillna(0)
    second_merge = first_merge.merge(users_time_feature, how='outer')
    third_merge = second_merge.merge(users_passed_course, how='outer').fillna(0)

    # Разделение на признаки и целевую переменную
    y = third_merge.passed_course.map(int)
    X = third_merge.drop('passed_course', axis=1)

    return X, y

def create_test_dataframe(events_data, submission_data):
    """
    Формирование тестового датасета

    Parameters:
    events_data: тестовые данные о событиях
    submission_data: тестовые данные о решениях

    Returns:
    X: матрица признаков для теста
    """
    # Фильтрация данных за два дня
    events_time_filter = time_filter(events_data)
    submissions_time_filter = time_filter(submission_data)

    # Создание таблиц с фичами
    users_data = data_preparation(events_time_filter, submissions_time_filter)
    users_time_feature = create_time_features(events_time_filter)
    users_steps_tried = steps_tried_feature(submissions_time_filter)
    users_data = correct_ratio_feature(users_data)

    # Объединение фич
    first_merge = users_data.merge(users_steps_tried, how='outer').fillna(0)
    X = first_merge.merge(users_time_feature, how='outer')

    return X

def random_forest_train(train_data, y, size=0.2):
    """
    Обучение модели Random Forest с подбором параметров

    Parameters:
    train_data: данные для обучения
    y: целевая переменная
    size: размер тестовой выборки

    Returns:
    best_model: лучшая модель
    """
    X_train, X_test, y_train, y_test = train_test_split(
        train_data, y, test_size=size, random_state=42
    )

    parameters = {
        'n_estimators': range(20, 51, 3),
        'max_depth': range(5, 14)
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, parameters, cv=5, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    print(f"Лучшие параметры: {grid_search.best_params_}")

    y_pred_prob = grid_search.predict_proba(X_test)
    roc_score = roc_auc_score(y_test, y_pred_prob[:, 1])
    score = grid_search.score(X_test, y_test)

    print(f"Правильность на тестовом наборе: {score:.3f}")
    print(f"ROC AUC score: {roc_score:.5f}")

    return grid_search.best_estimator_

def random_forest_test(train_data, y, test_data, size=0.2):
    """
    Финальное обучение и предсказание на тестовых данных

    Parameters:
    train_data: данные для обучения
    y: целевая переменная
    test_data: тестовые данные
    size: размер валидационной выборки
    """
    test_data = test_data.sort_values('user_id')
    best_model = random_forest_train(train_data, y, size)

    # Предсказание на тестовых данных
    y_pred_prob_final = best_model.predict_proba(test_data)

    # Создание файла с результатами
    result = test_data['user_id'].to_frame()
    result['is_gone'] = y_pred_prob_final[:, 1]

    # Сохранение результатов
    result[['user_id', 'is_gone']].to_csv('result.csv', index=False)
    print('Результаты сохранены в файл result.csv')

    return result

def main():
    """Основная функция выполнения pipeline"""
    print("Загрузка данных...")
    events_train, submissions_train, events_test, submissions_test = load_data()

    print("Подготовка данных для обучения...")
    X_train, y = create_dataframe(events_train, submissions_train)

    print("Подготовка тестовых данных...")
    X_test = create_test_dataframe(events_test, submissions_test)

    print("Обучение модели и предсказание...")
    results = random_forest_test(X_train, y, X_test)

    print("Готово!")

if __name__ == "__main__":
    main()