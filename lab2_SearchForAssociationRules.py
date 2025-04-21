import pandas as pd
import chardet
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import time
import matplotlib.pyplot as plt

with open('baskets.csv', 'rb') as f:
    result = chardet.detect(f.read())

df = pd.read_csv('baskets.csv', encoding=result['encoding'])
print(df.head())

def find_frequent_itemsets_and_association_rules(data, min_support, min_confidence, sort_by='support'):
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

    # Поиск ассоциативных правил
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules["antecedent"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
    rules["consequent"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))
    rules = rules[["antecedent", "consequent", "support", "confidence"]]

    # Сортировка результатов
    if sort_by == 'support':
        frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
        rules = rules.sort_values(by='support', ascending=False)
    elif sort_by == 'lexical':
        frequent_itemsets = frequent_itemsets.sort_values(by='itemsets')
        rules = rules.sort_values(by=['antecedent', 'consequent'])

    return frequent_itemsets, rules


def plot_performance_vs_confidence(data, min_support, min_confidence_values):
    # Список для хранения времени выполнения
    execution_times = []

    for min_confidence in min_confidence_values:
        start_time = time.time()
        _, rules = find_frequent_itemsets_and_association_rules(data, min_support, min_confidence)
        execution_time = time.time() - start_time
        execution_times.append(execution_time)

    # Создаем диаграмму
    plt.figure(figsize=(10, 6))
    plt.plot(min_confidence_values, execution_times, marker='o')
    plt.xlabel('Порог достоверности')
    plt.ylabel('Время выполнения (сек)')
    plt.title('Сравнение быстродействия поиска правил при изменении порога достоверности')
    plt.grid(True)
    plt.show()


def plot_rule_count_vs_confidence(data, min_support, min_confidence_values):
    # Список для хранения количества правил
    rule_counts = []

    for min_confidence in min_confidence_values:
        frequent_itemsets, rules = find_frequent_itemsets_and_association_rules(data, min_support, min_confidence)
        rule_counts.append(len(rules))

    # Создаем диаграмму
    plt.figure(figsize=(10, 6))
    plt.plot(min_confidence_values, rule_counts, marker='o', color='r')
    plt.xlabel('Порог достоверности')
    plt.ylabel('Количество правил')
    plt.title('Общее количество найденных правил при изменении порога достоверности')
    plt.grid(True)
    plt.show()


# Проведем эксперименты
min_support = 0.002
min_confidence_values = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

for min_confidence in min_confidence_values:
    frequent_itemsets, rules = find_frequent_itemsets_and_association_rules(df, min_support, min_confidence)

    # Вывод результатов
    print(f'\nПорог достоверности: {min_confidence}')
    print("\nЧастые наборы:")
    print(frequent_itemsets)
    print("\nАссоциативные правила:")
    print(rules)

# Настройки для экспериментов
min_support = 0.002
min_confidence_values = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# Построение графиков
plot_performance_vs_confidence(df, min_support, min_confidence_values)

plot_rule_count_vs_confidence(df, min_support, min_confidence_values)

# Настройки для экспериментов
min_support = 0.002
min_confidence = 0.65

# Найдем все частые наборы и ассоциативные правила
frequent_itemsets, rules = find_frequent_itemsets_and_association_rules(df, min_support, min_confidence)

# Отфильтруем правила, где количество элементов в антецеденте и консеквенте не превышает семь
filtered_rules = rules[
    (rules['antecedent'].apply(lambda x: len(x.split(', '))) +
     rules['consequent'].apply(lambda x: len(x.split(', '))) <= 7
     )]

# Выведем отфильтрованные правила
print("\nПравила, в которых антецедент и консеквент суммарно включают в себя не более семи объектов:")
print(filtered_rules)