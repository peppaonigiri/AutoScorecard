import scipy


def js_divergence(P, Q):
    M = (P+Q)/2
    return 0.5*scipy.stats.entropy(P, M)+0.5*scipy.stats.entropy(Q, M)


def symmetrical_kl_divergence(P, Q):
    return (scipy.stats.entropy(P, Q) + scipy.stats.entropy(Q, P)) / 2
