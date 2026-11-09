from .maxcut import MaxCutProblem
from .plot_matching import Buyer, Plot, PlotMatchingProblem
from .portfolio import PortfolioProblem
from .routing import RoutingProblem

# Approximate coordinates around Bhubaneswar-Cuttack, for demos only.
BBSR_CUTTACK_STOPS = {
    "Depot (Bhubaneswar, Master Canteen)": (20.2700, 85.8430),
    "Cuttack (Buxi Bazaar)": (20.4686, 85.8792),
    "Phulnakhara": (20.3900, 85.8600),
    "Patia": (20.3530, 85.8190),
    "Khandagiri": (20.2560, 85.7790),
}

__all__ = ["RoutingProblem", "PortfolioProblem", "PlotMatchingProblem", "MaxCutProblem", "Buyer",
           "Plot", "BBSR_CUTTACK_STOPS"]
