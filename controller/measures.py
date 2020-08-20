measure_details = [{'label': 'rank', 'value': 'rank', 'title':
    'SourceRank is the name for the algorithm that libraries.io uses to index search results.\n'
    'The maximum score for SourceRank is currently around 30 points. Their analysis is broken down into Code, '
    'Community, Distribution, Documentation and Usage'},
            {'label': 'stars', 'value': 'stars', 'title': 'How many ‘stars’ does the project have? As tagged by the community'},
            {'label': 'dependents_count', 'value': 'dependents_count', 'title': 'This is a number of dependent packages'},
            {'label': 'dependent_repos_count', 'value': 'dependent_repos_count', 'title': 'This is a number od dependent repositories'},
            {'label': 'forks', 'value': 'forks', 'title': 'This is a count of how many times has the package source been forked'}]

measures = ['rank', 'stars', 'dependents_count', 'dependent_repos_count', 'forks']


def get_measures():
    return measure_details
