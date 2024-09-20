import numpy as np


def run_binning_alg(scores, scores_1b, weights, weights_1b, 
                    num_bins=100000,
                    min_bkg_events=0.1,
                    min_stat_err=0.2):

    binning = [0]
    bin_yields = []
    sig_bin_yields = []
    acceptances = [0.5, 0.9, 0.99, 0.999]


    #Join the scores together, and find the marks above, WEIGHTED PROPERLY
    all_scores = np.append(scores, scores_1b)
    all_weights = np.append(weights, weights_1b)

    all_scores = np.clip(all_scores, a_min=0, a_max=1)

    fine_ad_bins = np.linspace(0,1,num_bins)

    hist, bin_edges = np.histogram(all_scores, bins=fine_ad_bins, weights=all_weights)

    cumsum = np.cumsum(hist)

    sum_w = sum(all_weights)
    print("Using total sum of weights: ", sum_w)

    #Get the stat errors
    yields_sqd, bin_edges_sqd = np.histogram(all_scores, bins=fine_ad_bins, weights=all_weights**2)
    cumsum_sqd = np.flip(np.cumsum(np.flip(yields_sqd)))   # Make sure the cumsum is done right to left


    cumsum_rightleft = np.flip(np.cumsum(np.flip(hist)))
    

    stat_err = np.sqrt(cumsum_sqd)
    frac_err = np.sqrt(cumsum_sqd) / abs(cumsum_rightleft)
    frac_err[np.isnan(frac_err)] = 0
    frac_err[np.isinf(frac_err)] = 0


    #Find the AD score value, where 90% of the background is accepted:
    for bkg_acceptance in acceptances:

        #Find the place where the cumsum equals the acceptance * total sum of weights

        #Take the first bin where the condition is met
        bin_pos = np.argwhere(cumsum >= bkg_acceptance*sum_w)[0]

        #Find the AD score point
        ad_score_val = fine_ad_bins[bin_pos]

        #Check if there's 0.1bkg event
        if sum_w - cumsum[bin_pos] > min_bkg_events:
            
            #Check that the fractional mc error is less than 20%:
            if frac_err[bin_pos] < min_stat_err:
                binning.append(ad_score_val[0])
                bin_yields.append(cumsum[bin_pos][0])
                sig_bin_yields.append((sum_w - cumsum[bin_pos])[0])

            else:
                print(f"Found {(1-bkg_acceptance)*100}% mark, but not met conditions of:\n   Min error < {min_stat_err}, actual err = {frac_err[bin_pos]}")

        else:
            print(f"Found {(1-bkg_acceptance)*100}% mark, but not met conditions of:\n   Min bkg > {min_bkg_events}, actual bkg = {sum_w-cumsum[bin_pos]}")

    binning.append(1)

    print("Got binnings: ", binning)
    print("With bkg bin yields of: ", bin_yields)
    print("And sig bin yields of: ", sig_bin_yields)

    output = {
        'binning': binning,
        'bin_yields' : bin_yields,
        'sig_bin_yields': sig_bin_yields
    }
    return output
    


