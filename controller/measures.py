measure_details = [{'label': 'stars', 'value': 'stars', 'title': 'How many ‘stars’ does the project have? As tagged by the community'},
            {'label': 'dependency_count', 'value': 'dependency_count', 'title': 'This is a number of dependencies'},
            {'label': 'dependents_count', 'value': 'dependents_count', 'title': 'This is a number of dependent packages'},
            {'label': 'dependent_repos_count', 'value': 'dependent_repos_count', 'title': 'This is a number of dependent repositories'},
            {'label': 'forks', 'value': 'forks', 'title': 'This is a count of how many times has the package source been forked'}]

measures = ['stars', 'dependency_count', 'dependents_count', 'dependent_repos_count', 'forks']


def get_measures():
    return measure_details
