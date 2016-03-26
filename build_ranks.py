#!/usr/env python3
# filename: build_ranks.py
# authors: Jon David and Jarrett Decker

import getopt
import sys
import numpy as np

from vocabulary import *
from ranking import *


def build(mlefile, mapfile, evidencefile, rankoutfile):
    """Builds a model and saves them in a file."""
    
    pstr = "\tBuilding word ranking..." + \
      "\n\t\tmleoutfile:{}" + \
      "\n\t\tmapoutfile:{}" + \
      "\n\t\tevidenceoutfile:{}" + \
      "\n\t\trankingoutfile:{}"
    print(pstr.format(mlefile, mapfile, evidencefile, rankoutfile))
    r = WordRanker(mlefile, mapfile, evidencefile)
    score_list = r.rank()

    rank_f = open(rankoutfile, 'w')
    for i in range(0, len(score_list)):
        print(score_list[i], file=rank_f)
    rank_f.close()
    

def main():
    """Defines the main function for this module."""

    if len(sys.argv) < 5:
        print("Must specify:")
        print("\t(1) [input] filename of MLE file")
        print("\t(2) [input] filename of MAP file")
        print("\t(3) [input] filename of EVIDENCE file")
        print("\t(3) [output] filename of ranking file")
        print("\nExiting...\n")
        sys.exit(2)
        
    mlefile = sys.argv[1]
    mapfile = sys.argv[2]
    evidencefile = sys.argv[3]
    rankoutfile = sys.argv[4]
    
    build(mlefile, mapfile, evidencefile, rankoutfile)
    
##==-- Main --==##
main()
