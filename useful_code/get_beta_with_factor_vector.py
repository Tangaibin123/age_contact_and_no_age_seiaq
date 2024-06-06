import get_eigenvalue
import get_age_effective_contact_matrix_with_factor_vector

def get_beta_with_factor_vector(R0, susceptibility_factor_vector, num_agebrackets, gamma_inverse, contact_matrix):
    """
    Get the transmissibility beta for an SIR model with an age dependent susceptibility drop factor and the age specific contact matrix.

    Args:
        R0 (basic reproduction number)            : the basic reproduction number
        susceptibility_factor_vector (np.ndarray) : vector of age specific susceptibility, where the value 1 means fully susceptibility and 0 means unsusceptible.
        num_agebrackets (int)                     : the number of age brackets of the contact matrix
        gamma_inverse (float)                     : the mean recovery period
        contact_matrix (np.ndarray)               : the contact matrix

    Returns:
        float: The transmissibility beta for an SIR compartmental model with an age dependent susceptibility drop factor and age specific contact patterns.
    """
    gamma = 1./gamma_inverse
    effective_matrix = get_age_effective_contact_matrix_with_factor_vector(contact_matrix.T, susceptibility_factor_vector)
    eigenvalue = get_eigenvalue(effective_matrix)
    beta = R0 * gamma / eigenvalue
    return beta
