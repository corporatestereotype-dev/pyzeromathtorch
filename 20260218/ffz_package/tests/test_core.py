import unittest
from ffz_package.core.torsion import FFZTorsion

class TestCore(unittest.TestCase):
    def test_torsion(self):
        torsion = FFZTorsion(1, 2)
        self.assertEqual(torsion.compute_torsion(), 1/2)
# Add more tests
