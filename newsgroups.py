#!/usr/env python3
# filename: newsgroups.py
# authors: Jon David and Jarrett Decker
# description:
#   Loads Newsgroups from file
#

class NewsGroups:
    DEFAULT_NEWSGROUPS_FILE = "./data/newsgrouplabels.txt"

    def __init__(self, filename=None):
        """Initializes DocVocabulary."""
        self.filename = filename
        if not self.filename:
            self.filename = self.DEFAULT_NEWSGROUPS_FILE
        self.group_to_id_dict =  {}
        self.id_to_group_dict = {}
        self.size = 0
        self.load()

    def load(self, filename=None):
        """Loads news groups from file."""
        if not filename:
            filename = self.DEFAULT_NEWSGROUPS_FILE
        f = open(filename, 'r')
        newsgroup_id = 1
        for line in f:
            newsgroup = line.strip().lower()
            self.id_to_group_dict[newsgroup_id] = newsgroup
            self.group_to_id_dict[newsgroup] = newsgroup_id
            newsgroup_id += 1
        self.size = len(self.id_to_group_dict)

    def get_id(self, newsgroup):
        """Returns the newsgroup_id associated with newsgroup."""
        return self.group_to_id_dict[newsgroup]

    def get_newsgroup(self, newsgroup_id):
        """Returns the group associated with newsgroup_id."""
        return self.id_to_group_dict[newsgroup_id]
