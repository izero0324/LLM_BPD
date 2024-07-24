import os
import subprocess
import tempfile

def check_pylint(code):
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        func_code = code
        f.write(f"{func_code}")
        temp_filename = f.name
    result = subprocess.run(['pylint', temp_filename, '--disable=missing-docstring,invalid-name'], capture_output=True, text=True, timeout=3)
    context = result.stdout.split('-----')[0]
    context = context.split(f'{temp_filename}:')[1:]
    score = result.returncode
    print(score)
    print(context)
    os.remove(temp_filename)
    return score, context

# check_pylint('''from collections import Counter
# def count_common(words):
#     word_counts = Counter(words)
#     return word_counts.most_common(4)
# ''')