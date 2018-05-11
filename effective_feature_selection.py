from sklearn.metrics import normalized_mutual_info_score as mi

def algo(df , n, chi, gain_ratio, info_gain, symmetric_uncertainty):
    F = []
    temp = []
    alpha = 0.75
    j = 0

    chi = chi.tolist()
    gain_ratio = gain_ratio.tolist()
    info_gain = info_gain.tolist()
    symmetric_uncertainty = symmetric_uncertainty.tolist()

    while (j < n):
        if (chi[0] == gain_ratio[0] == info_gain[0] == symmetric_uncertainty[0]):
            temp = chi[0]
            F.append(temp)
            j += 1
            chi.remove(temp)
            gain_ratio.remove(temp)
            info_gain.remove(temp)
            symmetric_uncertainty.remove(temp)

        else:
            mi_chi = mi(df.iloc[:, chi[0]], df.type)
            mi_gain_ratio = mi(df.iloc[:, gain_ratio[0]], df.type)
            mi_info_gain = mi(df.iloc[:, info_gain[0]], df.type)
            mi_symmetric_uncertainty = mi(df.iloc[:, symmetric_uncertainty[0]], df.type)
            max_value = max(mi_chi, mi_gain_ratio, mi_info_gain, mi_symmetric_uncertainty)

            if (max_value == mi_chi):
                temp = chi[0]
            elif (max_value == mi_gain_ratio):
                temp = gain_ratio[0]
            elif (max_value == mi_info_gain):
                temp = info_gain[0]
            elif (max_value == mi_symmetric_uncertainty):
                temp = symmetric_uncertainty[0]

            if not F:
                F.append(temp)
                j += 1
                chi.remove(temp)
                gain_ratio.remove(temp)
                info_gain.remove(temp)
                symmetric_uncertainty.remove(temp)

            else:
                if (any(mi(df.iloc[:, item], df.iloc[:, temp]) < alpha for item in F)):
                    F.append(temp)
                    j += 1
                    chi.remove(temp)
                    gain_ratio.remove(temp)
                    info_gain.remove(temp)
                    symmetric_uncertainty.remove(temp)

                else:
                    chi.remove(temp)
                    gain_ratio.remove(temp)
                    info_gain.remove(temp)
                    symmetric_uncertainty.remove(temp)
    return F