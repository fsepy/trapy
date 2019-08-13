# import required modules
import numpy as np
from trapy.func.view_factor import phi_parallel_any_br187 as phi_para
from trapy.func.view_factor import phi_perpendicular_any_br187 as phi_perp
from trapy.func.view_factor import fire_height_drysdale

# Helper functions
def calc_fire_diameter(Q_kW, Q_kW_m2):
    A = Q_kW / Q_kW_m2
    r = np.sqrt(A / np.pi)
    D = 2 * r
    return D


def calc_rad(plane_para, plane_perp, r_para, r_perp, heat_flux_rad):
    phi1 = phi_para(*plane_para, *r_para)
    phi2 = phi_perp(*plane_perp, *r_perp)

    r1 = phi1 * heat_flux_rad
    r2 = phi2 * heat_flux_rad
    return r1, r2


def calc_recepiant_heat_flux(Q, Q_pua, S, chi, h_r):
    D = calc_fire_diameter(Q, Q_pua)
    l = fire_height_drysdale(Q, D)

    # work out parallel plane dimensions and receiver location
    plane_para = (D, l)  # width, height
    r_para = (D / 2, h_r, S)  # x, y, S

    # work out perpendicular plane dimensions and receiver location
    plane_perp = (D, D)  # width, height
    r_perp = (D / 2, -S, abs(h_r - l))  # x, y, S

    # work out emitter radiation heat flux
    Q_r = chi * Q
    Q_r_flux = Q_r / (2 * D * l + np.pi * D ** 2 / 4)
    print(Q_r_flux)

    # work out receipiant rediation
    r1, r2 = calc_rad(plane_para, plane_perp, r_para, r_perp, Q_r_flux)

    return r1, r2


if __name__ == '__main__':
    pass
