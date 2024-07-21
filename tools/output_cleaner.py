def python_output(result):
    result = result.replace("```python",'')
    result = result.replace("```",'')
    result = result.replace("'''",'')
    result = result.replace('\n\n\n','')
    result = result.replace('"""','')
    return result

