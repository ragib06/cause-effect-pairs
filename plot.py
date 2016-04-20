__author__ = 'ragib'

import data_io
import sys, getopt
import numpy as np
import matplotlib.pyplot as plt

def main():

    numRows = 10
    targetVal = 0

    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:t:h")
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(2)

    clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
    for o, a in opts:
        if o == "-n":
            numRows = clamp(int(a), 1, 4050)
        elif o == "-t":
            targetVal = int(a)
        else:
            print "try help: python train.py -h"
            sys.exit(2)

    print "Reading " + str(numRows) + " rows in the training data"
    train = data_io.read_train_pairs(numRows)
    target = data_io.read_train_target(numRows)

    train.A = train.A.div(train.A.apply(max) - train.A.apply(min))
    train.B = train.B.div(train.B.apply(max) - train.B.apply(min))

    train = train.convert_objects(convert_numeric=True)
    # train = train.to_numeric()

    for i in range(1, numRows):
        if target.Target[i] == targetVal:
            A = train.iloc[i, :].A
            B = train.iloc[i, :].B

            plt.figure(i)
            plt.plot(range(len(A)), A)
            plt.plot(range(len(B)), B)
            plt.savefig('plots/' + str(targetVal) + '_' + str(i) + '.png')


    # train = train.astype(float)
    #
    # print train.info()





if __name__=="__main__":
    main()