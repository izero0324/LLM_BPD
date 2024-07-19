def count_common(words):
    counts = Counter(words)
    return counts.most_common(4)