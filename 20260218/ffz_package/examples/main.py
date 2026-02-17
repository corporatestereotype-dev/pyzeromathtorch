from ffz_package.core import *
from ffz_package.utils import *

def main():
    logger = setup_logger()
    logger.info("Running FFZ Example")
    torsion = FFZTorsion('a', 'b')
    print(torsion.compute_torsion())
    # Run all modules' examples
    # Auto-plot gradients, etc.
if __name__ == "__main__":
    main()
