measure_details = [{'label': 'stars', 'value': 'stars', 'title': 'How many ‘stars’ does the project have? As tagged by the community'},
            {'label': 'dependents_count', 'value': 'dependents_count', 'title': 'This is a number of dependent packages'},
            {'label': 'dependent_repos_count', 'value': 'dependent_repos_count', 'title': 'This is a number od dependent repositories'},
            {'label': 'forks', 'value': 'forks', 'title': 'This is a count of how many times has the package source been forked'}]

measures = ['stars', 'dependents_count', 'dependent_repos_count', 'forks']


def get_measures():
    return measure_details
