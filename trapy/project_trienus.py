# import required modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from trapy.func.view_factor import phi_parallel_any_br187 as phi_para
from trapy.func.view_factor import phi_perpendicular_any_br187 as phi_perp
from trapy.func.view_factor import fire_height_drysdale

sns.set_style('ticks', {'axis.grid': True,})

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


def run():
    # Define required variables
    Q_pua = 550  # HRR per unit area, in kW/m2
    chi = 0.5  # Radiation HRR portion of total HRR, dimensionless
    Qflux_r_d_rcp = 8  # Acceptable receipiant radiation heat flux, in kW/m2
    receiver_height = 4.35  # i.e. bottom edge of window, in m

    # Data containers
    list_S = np.arange(0.1, 10.1, 0.1)
    list_Q = []
    list_Qflux_r_rcp = []
    list_D = []
    list_l = []
    list_r1 = []
    list_r2 = []
    for S in list_S:

        Qflux_r_u = 50000
        Qflux_r_l = 100

        while True:
            r1, r2 = calc_recepiant_heat_flux(
                Q=np.average([Qflux_r_u, Qflux_r_l]),
                Q_pua=Q_pua,
                S=S,
                chi=chi,
                h_r=receiver_height
            )

            Qflux_r_rcp = np.sum([r1, r2])

            if abs(Qflux_r_rcp - Qflux_r_d_rcp) <= 0.01:
                break
            elif Qflux_r_rcp > Qflux_r_d_rcp:
                Qflux_r_u = np.average([Qflux_r_u, Qflux_r_l])
            elif Qflux_r_rcp < Qflux_r_d_rcp:
                Qflux_r_l = np.average([Qflux_r_u, Qflux_r_l])

        list_r1.append(r1)
        list_r2.append(r2)
        list_Q.append(np.average([Qflux_r_u, Qflux_r_l]))

    with open('test.csv', 'w') as f:
        s = ['{}, {}, {}, {}'.format(S, list_Q[i], list_r1[i], list_r2[i]) for i,S in enumerate(list_S)]
        s.insert(0, 'S [m], Q [kW], Q1 [kW], Q2 [kW]')

        f.write('\n'.join(s))

    fig, ax = plt.subplots(figsize=(3.94, 2.76))
    ax.plot(list_S, [i/1000. for i in list_Q])

    ax.set_ylabel('Permissible fire load [MW]')
    ax.set_xlabel('Separation distance between combustibles\n and wall with windows [m]')
    ax.set_xticks(ticks=np.arange(0, 10 + 0.001, 1))
    ax.set_yticks(ticks=np.arange(0, 20 + 0.001, 2))
    ax.grid(color='grey', linestyle='--', linewidth=.5)
    ax.legend().set_visible(False)
    # ax.legend(prop={'size': 7})
    fig.tight_layout()
    plt.show()
    fig.savefig(
        'test.png',
        transparent=True,
        bbox_inches='tight',
        dpi=300
    )


def run_check():
    # Define required variables
    Q_pua = 550  # HRR per unit area, in kW/m2
    chi = 0.5  # Radiation HRR portion of total HRR, dimensionless
    Q_r_d_rcp = 8  # Acceptable receipiant radiation heat flux, in kW/m2
    receiver_height = 4.35  # i.e. bottom edge of window, in m

    # Data containers
    list_separation = np.arange(0.1, 10.1, 0.1)
    list_Q_r_flux1 = []
    list_Q_r_flux2 = []
    list_D = []
    list_l = []
    list_r1 = []
    list_r2 = []
    for S in list_separation:
        Q_flux_r = calc_recepiant_heat_flux(Q=9500, Q_pua=Q_pua, S=S, chi=chi, h_r=receiver_height)
        list_r1.append(Q_flux_r[0])
        list_r2.append(Q_flux_r[1])
        list_Q_r_flux2.append(np.sum(Q_flux_r))

    fig, ax = plt.subplots()
    plt.plot(list_separation, list_Q_r_flux2)
    plt.plot(list_separation, list_r1)
    plt.plot(list_separation, list_r2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
        run()
