#!/usr/bin/env python3
"""A module implementing quaternions"""

import math
import numpy as np

class Quaternion:
    """Implementation of generic quaternions. If you're using quaternions to represent 3D rotations, use a
    UnitQuaternion object."""
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            assert args[0].shape in ((4,), (4, 1), (1, 4))
            self.q = args[0].copy()
        elif len(args) == 1 and isinstance(args[0], Quaternion):
            self.q = args[0].q.copy()
        elif len(args) == 2 and isinstance(args[0], (int, float)) and isinstance(args[1], np.ndarray):
            self.q = np.array([args[0], *args[1]])
        elif len(args) == 4:
            self.q = np.array([*args])

    @property
    def w(self):
        return self.q[0]

    @property
    def x(self):
        return self.q[1]

    @property
    def y(self):
        return self.q[2]

    @property
    def z(self):
        return self.q[3]

    @property
    def v(self):
        return self.q[1:]

    def __abs__(self):
        return self.w

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return self.q * other.q
        if isinstance(other, (int, float)):
            return Quaternion(other * self.q)
        return self.q * other

    def __rmul__(self, other):
        if isinstance(other, Quaternion):
            return other.q * self.q
        if isinstance(other, (int, float)):
            return Quaternion(other * self.q)
        return other * self.q

    def __matmul__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion._qmul(self, other)
        if isinstance(other, np.ndarray):
            if other.shape not in ((3,), (3, 1)):
                raise ValueError("A quaternion can only be multiplied by a vector of length 3 (shape (3,) or (3, 1))")
            q_r = Quaternion(0, *other.q)
            return Quaternion._qmul(self, q_r).v
        raise ValueError(f"I can't multiply a quaternion by this type, {type(other)}")

    def __add__(self, other):
        return Quaternion(self.q + other.q)

    def __sub__(self, other):
        return Quaternion(self.q - other.q)

    def __neg__(self):
        return Quaternion(-self.q)

    def __inv__(self):
        return self.inv()

    def __pow__(self, t):
        return Quaternion.exp(t * Quaternion.log(self))

    def __str__(self):
        return f"{self.w} + {self.x}i + {self.y}j + {self.z}k"

    def __repr__(self):
        return f"Quaternion({self.q})"

    @staticmethod
    def _qmul(q_l, q_r):
        """Perform quaternion multiplication"""
        return Quaternion(
            q_l.w*q_r.w - q_l.x*q_r.x - q_l.y*q_r.y - q_l.z*q_r.z,
            q_l.w*q_r.x + q_l.x*q_r.w + q_l.y*q_r.z - q_l.z*q_r.y,
            q_l.w*q_r.y + q_l.y*q_r.w + q_l.z*q_r.x - q_l.x*q_r.z,
            q_l.w*q_r.z + q_l.z*q_r.w + q_l.x*q_r.y - q_l.y*q_r.x,
            )

    def conj(self):
        return Quaternion(self.w, -self.v)

    def inv(self):
        qnorm2 = np.linalg.norm(self.q)**2
        return Quaternion(1/qnorm2 * self.q)

    @staticmethod
    def exp(q):
        theta = np.linalg.norm(q.v)
        v = q.v / theta
        return Quaternion(math.exp(q.q[0]) * math.cos(theta), math.exp(q.q[0]) * math.sin(theta) * v)

    @staticmethod
    def log(q):
        qnorm = np.linalg.norm(q.q)
        vnorm = np.linalg.norm(q.v)
        return Quaternion(math.log(qnorm), 1/vnorm * math.acos(q.w / qnorm) * q.v)

class UnitQuaternion(Quaternion):
    """Quaternion with unit norm, useful for representing 3D rotations"""
    def __init__(self, *args):
        super().__init__(*args)
        self.renormalize()

    def renormalize(self):
        self.q = self.q / np.linalg.norm(self.q)

    def inv(self):
        return self.conj()
