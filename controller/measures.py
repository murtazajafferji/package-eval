measure_details = [{'label': 'Stars', 'value': 'stars', 'title': 'How many ‘stars’ does the project have? As tagged by the community'},
{'label': 'Contributions', 'value': 'contributions_count', 'title': 'Number of unique contributors that have committed to the default branch.'},
{'label': 'Subscribers', 'value': 'subscribers_count', 'title': 'Number of subscribers to all notifications for the repository, only available for GitHub and Bitbucket.'},
{'label': 'Dependencies', 'value': 'dependency_count', 'title': 'This is a number of dependencies'},
{'label': 'Dependent Repos', 'value': 'dependent_repos_count', 'title': 'This is a number of dependent repositories'},
{'label': 'Dependents', 'value': 'dependents_count', 'title': 'This is a number of dependent packages'},
{'label': 'Forks', 'value': 'forks', 'title': 'This is a count of how many times has the package source been forked'},
{'label': 'Rank', 'value': 'rank', 'title':
    'SourceRank is the name for the algorithm that libraries.io uses to index search results.\n'
    'The maximum score for SourceRank is currently around 30 points. Their analysis is broken down into Code, '
    'Community, Distribution, Documentation and Usage'},
{'label': 'Size', 'value': 'size', 'title': 'Size of the repository in kilobytes, only available for GitHub and Bitbucket.'},
{'label': 'Open Issues', 'value': 'open_issues_count', 'title': 'Number of open issues on the repository bug tracker, only available for GitHub and GitLab.'},
{'label': 'Stargazers', 'value': 'stargazers_count', 'title': 'How many ‘stars’ does the project have? As tagged by the community'}]

measures = ['stars', 'contributions_count', 'subscribers_count', 'dependency_count', 'dependent_repos_count', 'dependents_count', 'forks', 'rank', 'size', 'open_issues_count', 'stargazers_count']

def get_measures():
    return measure_details

def find_measure_details_by_name(measure_name):
    return next((i for i in measure_details if i['value'] == measure_name), None)
