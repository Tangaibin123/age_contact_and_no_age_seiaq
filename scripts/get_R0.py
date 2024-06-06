from useful_code import get_age_effective_contact_matrix_with_factor_vector
from useful_code import get_eigenvalue
from useful_code import get_ages
import numpy as np
import os
import pandas as pd

# path to the directory where this script lives
thisdir = os.path.abspath('')

# path to the main directory of the repository
maindir = os.path.split(thisdir)[0]

# path to the results subdirectory
resultsdir = os.path.join(os.path.split(thisdir)[0], 'results')

# path to the data subdirectory
datadir = os.path.join(maindir, 'data')

def get_ages(location, country, level, num_agebrackets=85):
    """
    Get the age count for the synthetic population of the location.

    Args:
        location (str)        : name of the location
        country (str)         : name of the country
        level (str)           : name of level (country or subnational)
        num_agebrackets (int) : the number of age brackets

    Returns:
        dict: A dictionary of the age count.
    """
    if country == location:
        level = 'country'

    if level == 'country':
        file_name = country + '_' + level + '_level_age_distribution_' + '%i' % num_agebrackets + '.csv'
    else:
        file_name = country + '_' + level + '_' + location + '_age_distribution_' + '%i' % num_agebrackets + '.csv'
    file_path = os.path.join(datadir, 'origin_resource', file_name)
    df = pd.read_csv(file_path, delimiter=',', header=None)
    df.columns = ['age', 'age_count']
    ages = dict(zip(df.age.values.astype(int), df.age_count.values))
    return ages
num_agebrackets = 85
location = 'Shanghai'
country = 'China'
level = 'subnational'

ages = get_ages(location, country, level, num_agebrackets)
def get_R0_with_factor_vector(betaN, betaV, a, b1, b2, sigmaN_inverse, sigmaV_inverse, alpha_inverse, gam, susceptibility_factor_vector, num_agebrackets,
                              contact_matrix, i):
    """
    Get the basic reproduction number R0 for a SIR compartmental model with an age dependent susceptibility drop factor and the age specific contact matrix.

    Args:
        beta (float)                              : the transmissibility
        susceptibility_factor_vector (np.ndarray) : vector of age specific susceptibility, where the value 1 means fully susceptibility and 0 means unsusceptible.
        num_agebrackets (int)                     : the number of age brackets of the contact matrix
        gamma_inverse (float)                     : the mean recovery period
        contact_matrix (np.ndarray)               : the contact matrix

    Returns:
        float: The basic reproduction number for a SEAIQ compartmental model with an age dependent susceptibility drop factor and age specific contact patterns.
    """
    sigmaN = 1.0 / sigmaN_inverse
    sigmaV = 1.0 / sigmaV_inverse
    alpha = 1.0 / alpha_inverse
    effective_matrix = get_age_effective_contact_matrix_with_factor_vector(contact_matrix.T,susceptibility_factor_vector)
    eigenvalue = get_eigenvalue(effective_matrix)
    print('接触矩阵的最大特征值为：', eigenvalue)

    M = np.ones((num_agebrackets, 1))
    for j in range(num_agebrackets):
        M[i] = contact_matrix[i][j](ages[i]/ages[j])
    Ri = (M[i]*a*betaN)/alpha - (M[i]*betaV*(a - 1))/alpha + (M[i]*betaV*(a - 1)*(b2 - 1))/(b2*gam) - (M[i]*a*betaN*(b1 - 1))/(b1*gam)
    return Ri

