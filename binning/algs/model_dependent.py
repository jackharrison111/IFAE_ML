import numpy as np
import matplotlib.pyplot as plt

def run_modeldep_alg(scores, weights, 
                    num_bins=100000,
                    min_bkg_events=0.1,
                    min_stat_err=0.2,
                    yield_ratio=4):

    binning = [0]
    bin_yields = []
    
    #Make the overflow and underflow so that range is between 0,1
    all_scores = np.clip(scores, a_min=0, a_max=1)
    all_weights = weights

    #Get a very fine binning so that the approximation is as smooth as possible
    fine_ad_bins = np.linspace(0,1,num_bins)

    hist, bin_edges = np.histogram(all_scores, bins=fine_ad_bins, weights=all_weights)
    cumsum = np.flip(np.cumsum(np.flip(hist)))  #Make sure it goes right to left


    #Get the stat errors
    yields_sqd, bin_edges_sqd = np.histogram(all_scores, bins=fine_ad_bins, weights=all_weights**2)
    cumsum_sqd = np.flip(np.cumsum(np.flip(yields_sqd))) 


    stat_err = np.sqrt(cumsum_sqd)
    frac_err = np.sqrt(cumsum_sqd) / abs(cumsum)
    frac_err[np.isnan(frac_err)] = 0  # Set zero error to empty bins
    frac_err[np.isinf(frac_err)] = 0

    plt.figure()
    plt.plot(bin_edges[:-1], frac_err)
    plt.xlabel('Anomaly score')
    plt.ylabel(r'$\sigma_{w}$ / $\sum{w}$')
    plt.savefig("frac_error_2Z_0b.png")
    plt.close()

    plt.figure()
    plt.plot(bin_edges[:-1], cumsum)
    plt.xlabel('Anomaly score')
    plt.ylabel(r'Cum. Sum.')
    plt.savefig("cum_sum_2Z_0b.png")
    plt.close()

    plt.figure()
    plt.plot(bin_edges[70000:-1], cumsum[70000:])
    plt.xlabel('Anomaly score')
    plt.ylabel(r'Cum. Sum.')
    plt.savefig("cum_sum_shift_2Z_0b.png")
    plt.close()

    #Save a plot of frac_error


    binnings = []
    indices = []
    bin_yields = []
    
    #Get the first bin position
    #Again try changing to right most point
    #yield_index = np.argwhere(cumsum >= min_bkg_events)[-1]
    yield_index = np.argwhere(cumsum < min_bkg_events)[0]
    yield_val = fine_ad_bins[yield_index]
    print("Yield val: ", yield_val)

    #Try changing to the left-most point
    #error_index = np.argwhere(frac_err < min_stat_err)[-1]

    error_index = np.argwhere(frac_err > min_stat_err)
    if len(error_index) == 0:
        error_index = np.argwhere(frac_err < min_stat_err)[-1]
    else:
        error_index = error_index[0]
    error_val = fine_ad_bins[error_index]
    print("Error_val: ", error_val)
    
    if yield_val < error_val:
        binnings.append(yield_val)
        indices.append(yield_index)
    else:
        binnings.append(error_val)
        indices.append(error_index)


    #Now scan along until all of the yield has been used

    # Now increase the yields by the ratio each time
    prev_bin_yield = cumsum[indices[0]]
    bin_yields.append(prev_bin_yield)
    prev_bin_index = int(indices[0])

    #Set the new yield to search for
    new_yield = prev_bin_yield * yield_ratio


    #While there is still enough yield to make a next bin
    while max(cumsum) - cumsum[prev_bin_index] > new_yield:


        #Make a new cumsum starting at the right most bin
        new_cum_sum = np.flip(np.cumsum(np.flip(hist[:prev_bin_index])))
    
        # Get the position where the yield is over the new_yield threshold
        new_bin_index = int(np.argwhere(new_cum_sum > new_yield)[-1])


         #Save the ad position, and yield
        new_bin_val = fine_ad_bins[new_bin_index]
        new_bin_yield = new_cum_sum[new_bin_index]

        binnings.append(new_bin_val)
        bin_yields.append(new_bin_yield)


        new_yield = new_bin_yield * yield_ratio
        prev_bin_index = new_bin_index
        prev_bin_yield = new_bin_yield

        
    if 1 not in binnings:
        binnings = [1] + binnings
    if 0 not in binnings:
        binnings += [0]
    
    binnings.reverse()
    binnings = [float(b) for b in binnings]
    return binnings, bin_yields


