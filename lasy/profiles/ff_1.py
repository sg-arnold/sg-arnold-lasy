import numpy as np
from math import factorial
import math as m
from scipy.special import hermite
from scipy.constants import c, pi
from .profile import Profile
from .transverse import TransverseProfile

class FlyingFocus(Profile, TransverseProfile):
    def __init__(self, wavelength, pol, laser_energy, w0, tau, t_peak, beta_f=0, n_order=8, n_x=0, cep_phase=0, z_foc=0, v_foc=0):
        super().__init__(wavelength, pol)
        self.tau = tau
        self.t_peak = t_peak
        self.n_order = n_order
        self.cep_phase = cep_phase
        self.laser_energy = laser_energy
        self.lambda0 = wavelength
        self.w0 = w0
        self.beta_f = beta_f
        self.omega0 = 2 * pi * c / self.lambda0
        self.n_x = n_x
        self.z_foc = z_foc
        self.v_foc = v_foc

# y variable is omitted for this 2d version of the Hermite Gaussian mode
    def evaluate(self, x, y, t):
        # Term for wavefront curvature, waist and Gouy phase
        if self.z_foc and self.v_foc == 0:
            z_foc_over_zr = 0
        else:
            assert (
                self.lambda0 is not None
            ), "You need to pass the wavelength, when `z_foc` is non-zero."
            z_foc_over_zr = (self.lambda0 / (np.pi * self.w0**2)) * (self.z_foc + self.v_foc * (t - self.t_peak) /( 1 - self.beta_f))
        diffract_factor = 1.0 - 1j * z_foc_over_zr
        w = self.w0 * abs(diffract_factor)
        psi = np.angle(diffract_factor)
        #psi = m.atan(1.0 - 1j * z_foc_over_zr)
        
        # Hermite Gaussian transverse profile with coordinate transformation z_foc/z_r -> (z_foc - v_foc * t)/z_r
        envelope1 = (
            np.sqrt(2 / np.pi)
            * np.sqrt(1 / (2 ** (self.n_x) * factorial(self.n_x) * self.w0))
            * hermite(self.n_x)(np.sqrt(2) * x / w)
            * np.exp(
                -(x**2) / (self.w0**2 * diffract_factor)
                + 1.0j * self.n_x * psi
            )
            # Additional Gouy phase
            * np.sqrt(1.0 / diffract_factor)
        )

        # Ordinary Gaussian longitudinal profile
        envelope2 = np.exp(
        -np.power(((t - self.t_peak) ** 2) / self.tau**2, self.n_order / 2)
        + 1.0j * (self.cep_phase + self.omega0 * self.t_peak)
        )
        
        # Complete E field envelope
        envelope = envelope1 * envelope2

        return envelope
