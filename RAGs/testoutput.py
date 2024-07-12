def small_num(list1, n):
    """
    Find the n smallest numbers from a given list.

    Parameters
    ----------
    list1 : list
        Input list.
    n : int
        Number of smallest numbers to be returned.

    Returns
    -------
    list
        List of n smallest numbers.

    """
    import heapq
    return heapq.nsmallest(n, list1)

assert small_num([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],2)==[10,20]