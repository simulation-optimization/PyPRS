import math


def ZCDF(x):
    """
    Calculates the Cumulative Distribution Function (CDF) of the standard normal distribution.
    This uses a polynomial approximation.

    Args:
        x (float): The value at which to evaluate the CDF.

    Returns:
        float: The CDF value of the standard normal distribution at x.
    """
    neg = 1 if x < 0 else 0  # Flag to check if x is negative.
    if neg == 1:
        x *= -1  # Work with the absolute value of x.
    k = 1 / (1 + 0.2316419 * x)  # A factor used in the polynomial approximation.
    # The polynomial approximation for the tail probability.
    y = ((((1.330274429 * k - 1.821255978) * k + 1.781477937) * k - 0.356563782) * k + 0.319381530) * k
    # Combine with the standard normal PDF to get the final approximation.
    y = 1.0 - 0.3989422803 * math.exp(-0.5 * x * x) * y
    # Adjust the result based on the original sign of x. P(-x) = 1 - P(x).
    return (1 - neg) * y + neg * (1 - y)


def CHIPDF(N, C, lngam):
    """
    Calculates the Probability Density Function (PDF) of the chi-squared distribution.

    Args:
        N (int): The degrees of freedom.
        C (float): The chi-squared value (the point at which to evaluate the PDF).
        lngam (list): A pre-computed list of log-gamma values.

    Returns:
        float: The PDF value of the chi-squared distribution at C.
    """
    FLN2 = N / 2.0  # Calculate N/2.
    # Calculate the log of the PDF to avoid underflow/overflow with large numbers.
    TMP = -FLN2 * math.log(2) - lngam[N - 1] + (FLN2 - 1) * math.log(C) - C / 2.0
    # Return the exponential of the log-PDF to get the actual PDF value.
    return math.exp(TMP)


def rinott(T, PSTAR, NU):
    """
    Calculates the Rinott constant (h) for the Rinott selection procedure.

    This constant is found by solving an integral equation using Gauss-Hermite quadrature
    and a bisection search method to achieve the desired probability of correct selection (PSTAR).

    Args:
        T (int): The number of systems (alternatives) being compared.
        PSTAR (float): The desired probability of correct selection (e.g., 0.95).
        NU (int): The degrees of freedom, typically the initial sample size minus 1.

    Returns:
        float: The calculated Rinott constant, h.
    """
    # Initialize the logarithm of gamma values using pre-computed constants.
    LNGAM = [0.5723649429] + [0.0] * 49
    NU = min(NU, 50)  # Ensure NU does not exceed the pre-computed table size.
    WEX = [0.0] * 32  # Initialize weights for Gauss-Hermite quadrature.
    # Pre-defined weights for 32-point Gauss-Hermite quadrature.
    W = [0.10921834195238497114, 0.21044310793881323294, 0.23521322966984800539, 0.19590333597288104341,
         0.12998378628607176061, 0.70578623865717441560E-1, 0.31760912509175070306E-1, 0.11918214834838557057E-1,
         0.37388162946115247897E-2, 0.98080330661495513223E-3, 0.21486491880136418802E-3, 0.39203419679879472043E-4,
         0.59345416128686328784E-5, 0.74164045786675522191E-6, 0.76045678791207814811E-7, 0.63506022266258067424E-8,
         0.42813829710409288788E-9, 0.23058994918913360793E-10, 0.97993792887270940633E-12, 0.32378016577292664623E-13,
         0.81718234434207194332E-15, 0.15421338333938233722E-16, 0.21197922901636186120E-18, 0.20544296737880454267E-20,
         0.13469825866373951558E-22, 0.56612941303973593711E-25, 0.14185605454630369059E-27, 0.19133754944542243094E-30,
         0.11922487600982223565E-33, 0.26715112192401369860E-37, 0.13386169421062562827E-41, 0.45105361938989742322E-47]
    # Pre-defined abscissas (x-values) for 32-point Gauss-Hermite quadrature.
    X = [0.44489365833267018419E-1, 0.23452610951961853745, 0.57688462930188642649, 1.0724487538178176330,
         1.7224087764446454411, 2.5283367064257948811, 3.4922132730219944896, 4.6164567697497673878,
         5.9039585041742439466, 7.3581267331862411132, 8.9829409242125961034, 10.783018632539972068,
         12.763697986742725115, 14.931139755522557320, 17.292454336715314789, 19.855860940336054740,
         22.630889013196774489, 25.628636022459247767, 28.862101816323474744, 32.346629153964737003,
         36.100494805751973804, 40.145719771539441536, 44.509207995754937976, 49.224394987308639177,
         54.333721333396907333, 59.892509162134018196, 65.975377287935052797, 72.687628090662708639,
         80.187446977913523067, 88.735340417892398689, 98.829542868283972559, 111.75139809793769521]

    # Pre-compute the weighted exponential values (W * e^X) for efficiency in the quadrature loop.
    for I in range(1, 33):
        WEX[I - 1] = W[I - 1] * math.exp(X[I - 1])

    # Pre-compute the logarithm of gamma values using a recursive formula.
    for i in range(2, 26):
        LNGAM[2 * i - 2] = math.log(i - 1.5) + LNGAM[2 * i - 4]
        LNGAM[2 * i - 1] = math.log(i - 1.0) + LNGAM[2 * i - 3]

    # Initialize variables for the bisection search method.
    DUMMY = 1.0  # A scaling factor, set to 1.0 here.
    H = 4.0      # Initial guess for the Rinott constant h.
    LOWERH = 0.0 # Lower bound for h.
    UPPERH = 20.0# Upper bound for h.

    # Iterate to find the Rinott constant using the bisection method.
    for LOOPH in range(1, 51):
        ANS = 0.0 # This will hold the calculated probability for the current guess of H.
        # This double loop performs the 32x32 point Gauss-Hermite quadrature to solve the integral.
        for J in range(1, 33):
            TMP = 0.0
            for I in range(1, 33):
                # Calculate the inner part of the integral.
                TMP += WEX[I - 1] * ZCDF(H / math.sqrt(NU * (1 / X[I - 1] + 1 / X[J - 1]) / DUMMY)) \
                       * CHIPDF(NU, DUMMY * X[I - 1], LNGAM) * DUMMY
            # Raise the inner integral result to the power of (T-1).
            TMP = TMP ** (T - 1)
            # Add the contribution of this outer loop iteration to the final answer.
            ANS += WEX[J - 1] * TMP * CHIPDF(NU, DUMMY * X[J - 1], LNGAM) * DUMMY

        # Compare the calculated probability (ANS) with the desired probability (PSTAR).
        if abs(ANS - PSTAR) <= 0.000001:
            return H  # Return the constant h if the desired precision is achieved.
        elif ANS > PSTAR: # If the calculated probability is too high, H is too high.
            UPPERH = H
            H = (LOWERH + UPPERH) / 2.0  # Lower the upper bound and update H.
        else: # If the calculated probability is too low, H is too low.
            LOWERH = H
            H = (LOWERH + UPPERH) / 2.0  # Raise the lower bound and update H.
    return H  # Return the final value of H after the loop finishes.