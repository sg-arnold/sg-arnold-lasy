import numpy as np
from scipy.constants import c, pi
from .profile import Profile
from .transverse import TransverseProfile

class CustomCombinedProfile(Profile, TransverseProfile):
    r'''
    Class that expresses the gaussian profile without referring to longitudinal/transverse profiles.
    Acts as a child class to 'Profile' and 'TransverseProfile
    '''
    def __init__(self, wavelength, pol, laser_energy, w0, tau, t_peak, cep_phase=0, z_foc=0):
        super().__init__(wavelength, pol)
        self.lambda0 = wavelength
        self.omega0 = 2 * pi * c / self.lambda0
        self.w0 = w0
        self.laser_energy = laser_energy
        self.tau = tau
        self.t_peak = t_peak
        self.cep_phase = cep_phase
        if z_foc == 0:
            self.z_foc_over_zr = 0
        else:
            assert (
                wavelength is not None
            ), "You need to pass the wavelength, when `z_foc` is non-zero."
            self.z_foc_over_zr = z_foc * wavelength / (np.pi * w0**2)
    
    '''
    Method with two separate functions for position & time

    def transeval(self, x, y):
        # Term for wavefront curvature + Gouy phase
        diffract_factor = 1.0 - 1j * self.z_foc_over_zr
        # Calculate the argument of the complex exponential
        exp_argument = -(x**2 + y**2) / (self.w0**2 * diffract_factor)
        # Get the transverse profile
        envelope1 = np.exp(exp_argument) / diffract_factor

        return envelope1
    
    def longeval(self, t):
        envelope2 = np.exp(
            -((t - self.t_peak) ** 2) / self.tau**2
            + 1.0j * (self.cep_phase + self.omega0 * self.t_peak)
        )

        return envelope2
    '''

    def evaluate(self, x, y, t):
        # Transverse Envelope
        diffract_factor = 1.0 - 1j * self.z_foc_over_zr
        exp_argument = -(x**2 + y**2) / (self.w0**2 * diffract_factor)
        envelope1 = np.exp(exp_argument) / diffract_factor

        # Longitudinal Envelope
        envelope2 = np.exp(
            -((t - self.t_peak) ** 2) / self.tau**2
            + 1.0j * (self.cep_phase + self.omega0 * self.t_peak)
        )

        # Combined Longitudinal & Transverse
        envelope = envelope2 * envelope1 

        return envelope