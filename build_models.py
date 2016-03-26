#!/usr/env python3
# filename: build_models.py
# authors: Jon David and Jarrett Decker

import getopt
import sys

from training import Trainer


def build(trainingdatafile, traininglabelfile,
          mleoutfile, mapoutfile, evidenceoutfile,
          stoplistfile=None, betavalue=None):
    """Builds a model and saves them in a file."""
    pstr = "\tBuilding model with stopfile:{}, beta:{}" + \
      "\n\t\tdatafile:{}\n\t\tlabelfile:{}" + \
      "\n\t\tmleout:{}\n\t\tmapout:{}" + \
      "\n\t\tevidenceout:{}"
    print(pstr.format(stoplistfile, betavalue,
                      trainingdatafile, traininglabelfile,
                      mleoutfile, mapoutfile, evidenceoutfile))
    t = Trainer(trainingdatafile, traininglabelfile)
    MLE_vec, MAP_matrix, EVIDENCE_vec = t.generate_model(betavalue)
    t.save_model(MLE_vec, MAP_matrix, EVIDENCE_vec,
                 mleoutfile, mapoutfile, evidenceoutfile)


def main():
    """Defines the main function for this module."""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hosbdlxye:v", ["help", "output=", "stoplist=", "beta=", "data=", "label=", "mle=", "map=", "evidence="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) #will print something like "option -a not recognized"
        usage()
        print("Exiting...")
        sys.exit(2)
        
    output = None
    verbose = False
    stoplist = None
    beta = None
    trainingdatafile = None
    traininglabelfile = None
    mleoutfile = None
    mapoutfile = None
    evidenceoutfile = None

    for o,a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-o", "--output"):
            output = a
        elif o in ("-s", "--stoplist"):
            stoplist = a
        elif o in ("-b", "--beta"):
            beta = float(a)
        elif o in ("-d", "--data"):
            trainingdatafile = a
        elif o in ("-l", "--label"):
            traininglabelfile = a
        elif o in ("-x", "--mle"):
            mleoutfile = a
        elif o in ("-y", "--map"):
            mapoutfile = a
        elif o in ("-e", "--evidence"):
            evidenceoutfile = a
        else:
            assert False, "unhandled option"

    if (trainingdatafile is not None) and (traininglabelfile is not None) and \
        (mleoutfile is not None) and (mapoutfile is not None) and \
        (evidenceoutfile is not None):
        build(trainingdatafile, traininglabelfile,
              mleoutfile, mapoutfile, evidenceoutfile,
              stoplistfile=stoplist, betavalue=beta)
        return

    # Otherwise, something went wrong.
    print("Must specify:")
    print("\t(1) [input] filename of training data")
    print("\t(2) [input] filename of training labels")
    print("\t(3) [output] filename of MLE file")
    print("\t(4) [output] filename of MAP file")
    print("\t(5) [output] filename of EVIDENCE file")
    print("\nExample:")
    print("\tpython3 [options] build_predictions.py <data> <labels>" + \
          " <mle-file> <map-file> <evidence-file>")
    print("\nExiting.\n")
    sys.exit(2)


    
##==-- Main --==##
main()
