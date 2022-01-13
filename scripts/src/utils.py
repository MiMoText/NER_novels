'''
Utility functions for the use with the NER.
'''

def nullable_number(string):
    '''Utility function used by the CLI argument parser.
    Allows positive numbers as `ints` and returns `None`
    in any other case.
    '''
    try:
        if not string or string == 'None' or int(string) <= 0:
            return None
        return int(string)
    except:
        return None