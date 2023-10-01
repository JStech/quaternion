#!/usr/bin/env python3
"""Unit tests for quaternion package"""

import unittest
import numpy as np
from quaternion import Quaternion, UnitQuaternion

class TestQuaternion(unittest.TestCase):
    """Testing Quaternion and UnitQuaternion classes"""
    def setUp(self):
        self.q_id = Quaternion(1, 0, 0, 0)
        self.q_i = Quaternion(0, 1, 0, 0)
        self.q_j = Quaternion(0, 0, 1, 0)
        self.q_k = Quaternion(0, 0, 0, 1)
        self.q1 = Quaternion(1, 2, 3, 4)
        self.q2 = Quaternion(1, -2, -3, -4)
        self.uq1 = UnitQuaternion(0.5, 0.5, 0.5, 0.5)
    def test_operators(self):
        self.assertTrue((self.q_id * self.q1 == np.array([1, 0, 0, 0])).all())
        self.assertTrue(np.allclose((3.5 * self.q1).q, np.array([3.5, 7, 10.5, 14])))
        self.assertTrue(np.allclose((self.q_i @ self.q_j @ self.q_k).q, np.array([-1, 0, 0, 0])))
        self.assertTrue(np.allclose((self.q_i @ self.q_i).q, np.array([-1, 0, 0, 0])))
        self.assertTrue(np.allclose((self.q_j @ self.q_j).q, np.array([-1, 0, 0, 0])))
        self.assertTrue(np.allclose((self.q_k @ self.q_k).q, np.array([-1, 0, 0, 0])))
        self.assertTrue(np.allclose((self.q1 + self.q_k).q, np.array([1, 2, 3, 5])))
        self.assertTrue(np.allclose((self.q1**2).q, (self.q1 @ self.q1).q))
        self.assertTrue(np.allclose((self.uq1**3).q, -self.q_id.q))
        uq = UnitQuaternion(self.q1)
        self.assertTrue(np.allclose(uq.q, 30**-0.5 * np.array([1, 2, 3, 4])))

if __name__ == "__main__":
    unittest.main()
