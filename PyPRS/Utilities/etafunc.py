import math
import scipy.stats as stats


class EtaFunc:
    """
    A class to calculate the 'eta' parameter used in some screening procedures.

    This class contains methods to find the value of eta that satisfies a specific
    probability equation, either through a direct numerical integration or an
    approximation. It uses Gauss-Hermite quadrature for the integration and a
    bisection method to find the root of the equation.
    """
    # Flag to determine whether to use an approximate calculation for eta.
    APPROX_ETA = False

    # Array to store pre-computed weighted exponential values for quadrature.
    WEX = [0.0] * 32

    # Pre-defined weights for 32-point Gauss-Hermite quadrature.
    W = [
        0.10921834195238497114, 0.21044310793881323294, 0.23521322966984800539,
        0.19590333597288104341, 0.12998378628607176061, 0.70578623865717441560E-1,
        0.31760912509175070306E-1, 0.11918214834838557057E-1, 0.37388162946115247897E-2,
        0.98080330661495513223E-3, 0.21486491880136418802E-3, 0.39203419679879472043E-4,
        0.59345416128686328784E-5, 0.74164045786675522191E-6, 0.76045678791207814811E-7,
        0.63506022266258067424E-8, 0.42813829710409288788E-9, 0.23058994918913360793E-10,
        0.97993792887270940633E-12, 0.32378016577292664623E-13, 0.81718234434207194332E-15,
        0.15421338333938233722E-16, 0.21197922901636186120E-18, 0.20544296737880454267E-20,
        0.13469825866373951558E-22, 0.56612941303973593711E-25, 0.14185605454630369059E-27,
        0.19133754944542243094E-30, 0.11922487600982223565E-33, 0.26715112192401369860E-37,
        0.13386169421062562827E-41, 0.45105361938989742322E-47
    ]
    # Pre-defined abscissas (x-values) for 32-point Gauss-Hermite quadrature.
    X = [
        0.44489365833267018419E-1, 0.23452610951961853745, 0.57688462930188642649,
        1.0724487538178176330, 1.7224087764446454411, 2.5283367064257948811,
        3.4922132730219944896, 4.6164567697497673878, 5.9039585041742439466,
        7.3581267331862411132, 8.9829409242125961034, 10.783018632539972068,
        12.763697986742725115, 14.931139755522557320, 17.292454336715314789,
        19.855860940336054740, 22.630889013196774489, 25.628636022459247767,
        28.862101816323474744, 32.346629153964737003, 36.100494805751973804,
        40.145719771539441536, 44.509207995754937976, 49.224394987308639177,
        54.333721333396907333, 59.892509162134018196, 65.975377287935052797,
        72.687628090662708639, 80.187446977913523067, 88.735340417892398689,
        98.829542868283972559, 111.75139809793769521
    ]

    @staticmethod
    def eta_approx(x, n1, alpha1, k):
        """
        Calculates an approximation of the eta function value.
        This is a faster but less precise alternative to the integral method.

        Args:
            x (float): The current guess for eta.
            n1 (int): The Stage 1 sample size.
            alpha1 (float): The error rate for the screening.
            k (int): The number of systems (alternatives).

        Returns:
            float: The result of the approximation function. The goal is to find x
                   such that this function returns 0.
        """
        return 1. - math.pow(1.0 - alpha1, 1.0 / (k - 1.0)) - \
            2. * EtaFunc.la_gamma((n1 - 2.) / 2.) / \
            (EtaFunc.la_gamma((n1 - 1.) / 2.) * math.sqrt(math.pi) *
             x * math.pow(x * x + 1., (n1 - 2.0) / 2.))

    @staticmethod
    def eta_integral(x, n1, alpha1, k):
        """
        Calculates the eta function value using numerical integration.
        This method is more precise but computationally more intensive.

        Args:
            x (float): The current guess for eta.
            n1 (int): The Stage 1 sample size.
            alpha1 (float): The error rate for the screening.
            k (int): The number of systems (alternatives).

        Returns:
            float: The result of the integral function. The goal is to find x
                   such that this function returns 0.
        """
        # Right-hand side of the equation we are trying to solve.
        RHS = 1. - math.pow(1.0 - alpha1, 1.0 / (k - 1.0))
        # Left-hand side, which will be calculated via numerical integration.
        LHS = 0.0

        # Create statistical distribution objects for repeated use.
        normDist = stats.norm(0, 1) # Standard normal distribution.
        chi_sqDist = stats.chi2(n1 - 1) # Chi-squared distribution.

        # Calculate the LHS integral using 32-point Gauss-Hermite quadrature.
        for i in range(32):
            LHS += EtaFunc.WEX[i] * 4 * \
                   (1. - normDist.cdf(x * math.sqrt(EtaFunc.X[i]))) * \
                   chi_sqDist.pdf(EtaFunc.X[i]) * \
                   (1. - chi_sqDist.cdf(EtaFunc.X[i]))

        return RHS - LHS

    @staticmethod
    def eval_eta(x, n1, alpha1, k, f):
        """
        A helper function to evaluate a chosen eta function (approximate or integral).

        Args:
            x (float): The input value for the eta function.
            n1 (int): Stage 1 sample size.
            alpha1 (float): Error rate.
            k (int): Number of systems.
            f (callable): The specific eta function to use (eta_approx or eta_integral).

        Returns:
            float: The value of the chosen eta function.
        """
        return f(x, n1, alpha1, k)

    @staticmethod
    def la_gamma(x):
        """
        Calculates the logarithm of the gamma function using Lanczos approximation.

        Args:
            x (float): The input value.

        Returns:
            float: The natural logarithm of the gamma function of x.
        """
        # Coefficients for the Lanczos approximation.
        p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
             771.32342877765313, -176.61502916214059, 12.507343278686905,
             -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
        g = 7

        # Use the reflection formula for negative values.
        if x < 0.5:
            return math.pi / (math.sin(math.pi * x) * EtaFunc.la_gamma(1 - x))

        x -= 1
        a = p[0]
        t = x + g + 0.5

        # The main Lanczos approximation series.
        for i in range(1, 9):
            a += p[i] / (x + i)

        return math.sqrt(2 * math.pi) * math.pow(t, x + 0.5) * math.exp(-t) * a

    @staticmethod
    def find_eta(n1, alpha1, nSys):
        """
        Finds the root of the eta function using the bisection method.

        Args:
            n1 (int): The Stage 1 sample size.
            alpha1 (float): The error rate.
            nSys (int): The number of systems (alternatives).

        Returns:
            float: The calculated value of eta.
        """
        # Initialize the interval [a, b] for the bisection search.
        a = 0.0
        b = 15.0
        e = 0.0001  # Tolerance for convergence.
        dx = b - a
        x = dx / 2.0  # Initial guess is the midpoint.
        k = 0  # Iteration counter.

        # Pre-compute weighted exponential values for efficiency.
        for i in range(32):
            EtaFunc.WEX[i] = EtaFunc.W[i] * math.exp(EtaFunc.X[i])

        # Choose the eta function to use based on the class flag.
        f = EtaFunc.eta_approx if EtaFunc.APPROX_ETA else EtaFunc.eta_integral

        # Check if a root exists within the interval [a, b].
        # The function values at the endpoints must have opposite signs.
        if ((EtaFunc.eval_eta(a, n1, alpha1, nSys, f) >= 0 and EtaFunc.eval_eta(b, n1, alpha1, nSys, f) >= 0) or
                (EtaFunc.eval_eta(a, n1, alpha1, nSys, f) < 0 and EtaFunc.eval_eta(b, n1, alpha1, nSys, f) < 0)):
            print("\nThe values of f(a) and f(b) do not differ in sign.\n")
            return 0.0

        # Perform bisection method to find the root.
        while abs(dx) > e and k < 1000 and EtaFunc.eval_eta(x, n1, alpha1, nSys, f) != 0:
            x = (a + b) / 2.0
            # If the signs of f(a) and f(x) differ, the root is in [a, x].
            if (EtaFunc.eval_eta(a, n1, alpha1, nSys, f) * EtaFunc.eval_eta(x, n1, alpha1, nSys, f)) < 0:
                b = x
                dx = b - a
            # Otherwise, the root is in [x, b].
            else:
                a = x
                dx = b - a
            k += 1

        return x