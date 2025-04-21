import pandas as pd
import chardet
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import time
import matplotlib.pyplot as plt

with open('baskets.csv', 'rb') as f:
    result = chardet.detect(f.read())

df = pd.read_csv('baskets.csv', encoding=result['encoding'])
df

def find_frequent_itemsets(data, min_support, sort_by='support'):
    # Преобразование данных в список списков для подачи в метод fit
    transactions = data.values.tolist()

    # Удаление значений NaN из транзакций
    transactions = [[item for item in transaction if not pd.isnull(item)] for transaction in transactions]

    # Преобразование в разряженную матрицу
    te = TransactionEncoder()
    te_transform = te.fit(transactions).transform(transactions)
    transactions_matrix = pd.DataFrame(te_transform, columns=te.columns_)

    # Поиск частых наборов с помощью алгоритма Apriori
    frequent_itemsets = apriori(transactions_matrix, min_support=min_support, use_colnames=True)

    # Сортировка результатов
    if sort_by == 'support':
        frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
    elif sort_by == 'lexical':
        frequent_itemsets = frequent_itemsets.sort_values(by='itemsets')

    return frequent_itemsets

min_support = 0.01
result = find_frequent_itemsets(df, min_support, sort_by='support')
result.head(20)

# Минимальное число элементов в наборе - 2
result_with_2_items = result[result['itemsets'].apply(lambda x: len(x)) == 2]
result_with_2_items

# Минимальное число элементов в наборе - 3
result_with_3_items = result[result['itemsets'].apply(lambda x: len(x)) == 3]
result_with_3_items


# Измерение времени выполнения
def measure_execution_time(data, min_support):
    start_time = time.time()
    find_frequent_itemsets(data, min_support)
    end_time = time.time()
    return end_time - start_time


# Диаграмма сравнения времени выполнения
def plot_execution_time(data, min_support_values):
    execution_times = [measure_execution_time(data, min_support) for min_support in min_support_values]

    plt.figure(figsize=(10, 6))
    plt.plot(min_support_values, execution_times, marker='o', linestyle='-')
    plt.title('Сравнение времени выполнения при изменении порога поддержки')
    plt.xlabel('Порог поддержки')
    plt.ylabel('Время выполнения (секунды)')
    plt.grid(True)
    plt.show()


# Диаграмма количества частых наборов объектов различной длины
def plot_frequent_itemset_length(data, min_support_values):
    number_of_items_by_length = {i: [] for i in range(1, len(min_support_values) - 1)}
    supports = []

    for min_support in min_support_values:
        frequent_itemsets = find_frequent_itemsets(data, min_support)

        # Подсчет количества наборов разной длины
        counts = frequent_itemsets['itemsets'].apply(lambda x: len(x)).value_counts().sort_index()

        # Сохраняем результаты
        for length in number_of_items_by_length.keys():
            number_of_items_by_length[length].append(counts.get(length, 0))
        supports.append(min_support)

        # Создаем диаграмму
    plt.figure(figsize=(10, 6))

    # Для каждой длины набора
    for length, counts in number_of_items_by_length.items():
        plt.plot(supports, counts, label=f'Кол-во элементов в наборе: {length}')

    plt.xlabel('Поддержка')
    plt.ylabel('Число наборов')
    plt.title('Число частых наборов разных длин при изменяемом пороге поддержки')
    plt.legend()
    plt.grid(True)
    plt.show()


min_support_values = [0.01, 0.03, 0.05, 0.10, 0.15]

plot_execution_time(df, min_support_values)

plot_frequent_itemset_length(df, min_support_values)