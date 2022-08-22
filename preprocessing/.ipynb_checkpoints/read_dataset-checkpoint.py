

#Function to calculate the MC weights
def calculate_mcweight(df):
    
    total_lum = 138965.16
    df.loc[df['RunYear'].isin([2015,2016]), 'lumi_scale'] = 36207.66*(1/total_lum)
    df.loc[df['RunYear'].isin([2017]), 'lumi_scale'] = 44307.4*(1/total_lum)
    df.loc[df['RunYear'].isin([2018]), 'lumi_scale'] = 58450.1*(1/total_lum)

    df['weight'] = df['lumi_scale']*df['custTrigSF_TightElMediumMuID_FCLooseIso_DLT']*df['weight_pileup']*df['jvtSF_customOR']*df['bTagSF_weight_DL1r_77']*df['weight_mc']*df['xs']*df['lep_SF_CombinedLoose_0']*df['lep_SF_CombinedLoose_1']*df['lep_SF_CombinedLoose_2']*df['lep_SF_CombinedLoose_3']/df['totalEventsWeighted']

    return df


def get_dataset(infile, chosen_samples, )
    ...