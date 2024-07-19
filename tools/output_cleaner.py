def python_output(result):
    result = result.replace("```python",'')
    result = result.replace("```",'')
    result = result.replace("'''",'')
    result = result.replace('\n\n','')
    result = result.replace('"""','')
    result = result.replace("\begin{pre}", '')
    return result

