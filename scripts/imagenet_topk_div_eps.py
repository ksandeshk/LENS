# Obtains LENS and top-k-div with LENS results for explanation maps obtained for different epsilon values with different attacks.

import numpy as np 

# Obtains pixel indices around index given by x of window size 3x3
def submat3x3(x): 
    x1 = x + 227 
    x2 = x - 227     
    return [x2-1,x2,x2+1,x-1,x,x+1,x1-1,x1,x1+1]

# Obtains pixel indices around index given by x of window size 5x5
def submat5x5(x): 
    x1 = x + 227 
    x2 = x - 227     
    x3 = x + 454 
    x4 = x - 454     
    return [x2-2,x2-1,x2,x2+1,x2+2,x-2,x-1,x,x+1,x+2,x1-2,x1-1,x1,x1+1,x1+2,x3-2,x3-1,x3,x3+1,x3+2,x4-2,x4-1,x4,x4+1,x4+2]

# Obtains pixel indices around index given by x of window size 7x7
def submat7x7(x): 
    x1 = x + 227 
    x2 = x - 227     
    x3 = x + 454 
    x4 = x - 454     
    x5 = x + 681 
    x6 = x - 681     
    return [x2-3,x2-2,x2-1,x2,x2+1,x2+2,x2+3,x-3,x-2,x-1,x,x+1,x+2,x+3,x1-3,x1-2,x1-1,x1,x1+1,x1+2,x1+3,x3-3,x3-2,x3-1,x3,x3+1,x3+2,x3+3,x4-3,x4-2,x4-1,x4,x4+1,x4+2,x4+3,x5-3,x5-2,x5-1,x5,x5+1,x5+2,x5+3,x6-3,x6-2,x6-1,x6,x6+1,x6+2,x6+3]


train_type = ["NAT", "PGD"] # List of training types
lname = ["SG", "DL", "IG"] # List of explanation methods 
eps = [1,2,4,8] # List of epsilon values
attack_type = ["random", "mass_center", "topK"] # Attack types

# Add pixels of submat size around the pixel of index pixInd
def addwindowPx(pixInd):
    finalo = []
    finalo = finalo + submat(pixInd)

    finalo = np.array(finalo)
    finalo= finalo[finalo < imagedimflat]
    finalo= finalo[finalo > 0]
    fo = np.unique(np.array(finalo))
    return fo

# Add pixels of submat size around the list of pixel indices in inpList
def addwindowList(inpList):
    lfinalo = []
    for i in inpList:
        lfinalo = lfinalo + submat(i)

    lfinalo = np.array(lfinalo)
    lfinalo = lfinalo[lfinalo < imagedimflat]
    lfinalo = lfinalo[lfinalo > 0]
    lfo = np.unique(np.array(lfinalo))
    return lfo

# Obtain top-k-div k=eval_topk
def topkDiv(oList, eval_topk): #oList is sorted in descending order
    finalo = []
    finalo.append(oList[0])
    oList = np.setdiff1d(oList, addwindowPx(oList[0]), assume_unique=True)
    for i in range(eval_topk-1):
        oList = np.setdiff1d(oList, addwindowPx(oList[0]), assume_unique=True) 
        oList = oList[oList < imagedimflat]
        oList = oList[oList > 0]
        finalo.append(oList[0])
    finalo = np.array(finalo)
    finalo = finalo[finalo < imagedimflat]
    finalo = finalo[finalo > 0]
    fo = np.unique(np.array(finalo))
    return fo

for at in attack_type: # which attack type
    for e in eps: # epsilon value

        num_samples = # enter number of samples
        og = np.load("<path of maps of unperturbed image>") # path of npy of explanation maps of unperturbed samples
        pg = np.load("<path of maps of perturbed image>") # path of npy of explanation maps of perturbed samples

        for t in range(3): # assign different window size
            if t == 0:
                submat = submat3x3
                eval_topk1=6000 # heuristic value used

            if t == 1:
                submat = submat5x5
                eval_topk1=15000

            if t == 2:
                submat = submat7x7
                eval_topk1=25000

            sum_o_p = 0.0
            sum_fo_p = 0.0
            sum_o_fp = 0.0
            sum_fo_fp = 0.0
            sum_lfo_fp = 0.0
            sum_fo_lfp = 0.0

            eval_topk = 1000
            imagedimflat = 51529 #227*227 image size

            for k in range(0,num_samples):
                o = np.argsort(og[k])[-eval_topk:] # top-k
                p = np.argsort(pg[k])[-eval_topk:] # top-k
                o1 = np.argsort(og[k])[-eval_topk1:][::-1] # used for top-k-div
                p1 = np.argsort(pg[k])[-eval_topk1:][::-1]  # used for top-k-div

                ofo = addwindowList(o) # w-LENS-recall(top-k)
                ofp = addwindowList(p) # w-LENS-prec(top-k)

                fo = topkDiv(o1, eval_topk) # top-k-div of map of unperturbed image
                fp = topkDiv(p1, eval_topk) # top-k-div of map of perturbed image

                lfo = addwindowList(fo) # w-LENS-recall(top-k-div)
                lfp = addwindowList(fp) # w-LENS-prec(top-k-div)

                sum_o_p = sum_o_p + float(len(np.intersect1d(o,p)))/eval_topk 
                sum_fo_p = sum_fo_p + float(len(np.intersect1d(ofo,p)))/eval_topk 
                sum_o_fp = sum_o_fp + float(len(np.intersect1d(o,ofp)))/eval_topk 
                sum_fo_fp = sum_fo_fp + float(len(np.intersect1d(fo,fp)))/eval_topk 
                sum_lfo_fp = sum_lfo_fp + float(len(np.intersect1d(lfo,fp)))/eval_topk 
                sum_fo_lfp = sum_fo_lfp + float(len(np.intersect1d(fo,lfp)))/eval_topk 

            print("top-k intersection  : ",sum_o_p/num_samples)
            print("LENS-recall : ",sum_fo_p/num_samples)
            print("LENS-prec   : ",sum_o_fp/num_samples)
            print("top-k-div intersection  : ",sum_fo_fp/num_samples)
            print("LENS-recall-div : ",sum_lfo_fp/num_samples)
            print("LENS-prec-div   : ",sum_fo_lfp/num_samples)

