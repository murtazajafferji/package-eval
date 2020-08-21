measure_details = [{'label': 'stars', 'value': 'stars', 'title': 'How many ‘stars’ does the project have? As tagged by the community'},
{'label': 'stargazers_count', 'value': 'stargazers_count', 'title': 'How many ‘stars’ does the project have? As tagged by the community'},
{'label': 'contributions_count', 'value': 'contributions_count', 'title': 'How many ‘stars’ does the project have? As tagged by the community'},
{'label': 'subscribers_count', 'value': 'subscribers_count', 'title': 'How many ‘stars’ does the project have? As tagged by the community'},
            {'label': 'dependency_count', 'value': 'dependency_count', 'title': 'This is a number of dependencies'},
            {'label': 'dependents_count', 'value': 'dependents_count', 'title': 'This is a number of dependent packages'},
            {'label': 'dependent_repos_count', 'value': 'dependent_repos_count', 'title': 'This is a number of dependent repositories'},
            {'label': 'forks', 'value': 'forks', 'title': 'This is a count of how many times has the package source been forked'},
            {'label': 'rank', 'value': 'rank', 'title':
    'SourceRank is the name for the algorithm that libraries.io uses to index search results.\n'
    'The maximum score for SourceRank is currently around 30 points. Their analysis is broken down into Code, '
    'Community, Distribution, Documentation and Usage'},
    {'label': 'size', 'value': 'size', 'title': 'How many ‘stars’ does the project have? As tagged by the community'},
{'label': 'open_issues_count', 'value': 'open_issues_count', 'title': 'How many ‘stars’ does the project have? As tagged by the community'}]
measures = ['stars', 'stargazers_count', 'contributions_count', 'subscribers_count', 'dependency_count', 'dependents_count', 'dependent_repos_count', 'forks', 'rank', 'size', 'open_issues_count']


def get_measures():
    return measure_details
