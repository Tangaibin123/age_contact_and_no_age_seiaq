def get_age_effective_contact_matrix_with_factor_vector(contact_matrix, susceptibility_factor_vector):
    """
    Get an effective age specific contact matrix with an age dependent susceptibility drop factor.

    Args:
        contact_matrix (np.ndarray)        : the contact matrix
        susceptibility_factor_vector (int): vector of age specific susceptibility, where the value 1 means fully susceptibility and 0 means unsusceptible.


    Returns:
        np.ndarray: A numpy square matrix that gives the effective contact matrix given an age dependent susceptibility drop factor.
    """
    effective_matrix = contact_matrix * susceptibility_factor_vector
    return effective_matrix
