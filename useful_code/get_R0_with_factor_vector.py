import get_eigenvalue
import get_age_effective_contact_matrix_with_factor_vector

def get_R0_with_factor_vector(betaN, betaV, a, b1, b2, sigmaN1_inverse, sigmaN2_inverse, sigmaV1_inverse, sigmaV2_inverse, alpha_inverse, C, susceptibility_factor_vector, num_agebrackets, contact_matrix):
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
    sigmaN1 = 1.0 / sigmaN1_inverse
    sigmaN2 = 1.0 / sigmaN2_inverse
    sigmaV1 = 1.0 / sigmaV1_inverse
    sigmaV2 = 1.0 / sigmaV2_inverse
    alpha = 1.0 / alpha_inverse
    effective_matrix = get_age_effective_contact_matrix_with_factor_vector(contact_matrix.T, susceptibility_factor_vector)
    eigenvalue = get_eigenvalue(effective_matrix)
    R0 = ((betaV*sigmaV2*(1-a))((1-b2)*alpha + b2*C)/(C*alpha*(sigmaV1-b2*sigmaV1 + b2*sigmaV2)) + ((a*betaN)*(b1*sigmaN2*C+(1-b1)*sigmaN1*alpha))/(C*alpha*(sigmaN1-b1*sigmaN1+b1*sigmaN2)))*eigenvalue
    return R0
