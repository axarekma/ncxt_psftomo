import unittest
import ncxt_psftomo as module


class TestOpenMP(unittest.TestCase):
    def test_omp(self):
        self.assertEqual(module.test_omp() > 1, True)


if __name__ == '__main__':
    unittest.main()
