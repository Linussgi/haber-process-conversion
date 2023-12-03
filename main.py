import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")

N2_COEFFS_500K = {
    "A": 28.98641,
    "B": 1.853978,
    "C": -9.647459,
    "D": 16.63537,
    "E": 0.000117
    }

N2_COEFFS_2000K = {
    "A": 19.50583,
    "B": 19.88705,
    "C": -8.598535,
    "D": 1.369784,
    "E": 0.527601
    }

H2_COEFFS = {
    "A": 33.066178,
    "B": -11.363417,
    "C": 11.432816,
    "D": -2.772874,
    "E": -0.158558
    }

NH3_COEFFS = {
    "A": 19.99563,
    "B": 49.77119,
    "C": -15.37599,
    "D": 1.921168,
    "E": 0.189174
    }

DH_298 = -92.4 * 1000
DS_298 = -99
STD_TEMP = 298
N_N2 = 1

# Number of pressure values between 1atm and 200atm to test
PN = 75


def gibbs_calc(dh, ds, temp):
    return dh - temp * ds


def equi_constant_calc(dg, temp):
    return np.exp(-dg / (8.314 * temp))


def heat_cap_calc(A, B, C, D, E, temp):
    return (A / 1000) + (B * (temp) / 1000) + (C * (temp ** 2) / 1000) + (D * (temp ** 3) / 1000) + (1000 * E / temp ** 2)


def integrated_heat_cap(A, B, C, D, E, temp):
    return (A * temp / 1000) + (B * temp ** 2 / 2000) + (C * temp ** 3 / 3000) + (D * temp ** 4 / 4000) - (1000 * E / temp)


def vant_hoff_calc(temp, dH):
    return -1 * (dH) / (temp * 8.314)


def solve_equation(x, m, p):
    n2_frac = (1 - x) / (4 - 2 * x)
    h2_frac = (3 - 3 * x) / (4 - 2 * x)
    nh3_frac = (2 * x) / (4 - 2 * x)

    return np.exp(m * n2_frac) - (((nh3_frac ** 2) / (n2_frac * (h2_frac ** 3))) * (1 / (p ** 2)))


temp_range = np.linspace(300, 910, 100)  # In Kelvin
pressure_range = np.linspace(10, 200, PN)  # In atm

conversion_solutions = []

for pressure in pressure_range:
    current_pressure_solution = []

    for temperature in temp_range:
        dG_298 = gibbs_calc(DH_298, DS_298, temperature)

        if temperature < 500:
            n_2T = (1 / 1000) * (integrated_heat_cap(temp=STD_TEMP, **N2_COEFFS_500K) - integrated_heat_cap(temp=temperature, **N2_COEFFS_500K))
        else:
              n_2T = (1 / 1000) * (integrated_heat_cap(temp=STD_TEMP, **N2_COEFFS_2000K) - integrated_heat_cap(temp=temperature, **N2_COEFFS_2000K))

        h_2T = (1 / 1000) * (integrated_heat_cap(temp=STD_TEMP, **H2_COEFFS) - integrated_heat_cap(temp=temperature, **H2_COEFFS))
        nh_3T = (1 / 1000) * (-1 * integrated_heat_cap(temp=STD_TEMP, **NH3_COEFFS) + integrated_heat_cap(temp=temperature, **NH3_COEFFS))

        rxn_enthalpy = n_2T + h_2T + nh_3T + DH_298

        heat_term = vant_hoff_calc(temperature, rxn_enthalpy)

        thermo_term = (heat_term * N_N2 - (dG_298) / (298 * 8.314))

        initial_guess = 0.98
        solution = fsolve(solve_equation, initial_guess, args=(thermo_term, pressure))

        conversion = solution / N_N2
        current_pressure_solution.append(conversion)
    else:
        conversion_solutions.append(current_pressure_solution)

if len(pressure_range) < 8:  # Plot line graph
    fig, ax = plt.subplots(figsize=[12, 8])

    x_data = temp_range

    COLOURS = ["r", "g", "b", "c", "m", "y", "k"]
    lines = []

    for index, solution in enumerate(conversion_solutions):
        color = COLOURS[index]

        line = ax.errorbar(
        x_data,
        solution,
        yerr=None,
        fmt=f"o-{color}",
        label=f"{np.round(pressure_range[index], decimals=2)} atm",
        linewidth=1,
        elinewidth=2,
        capsize=2,
        capthick=2,
        ms=0
        )

        lines.append(line)

    ax.grid()
    ax.legend(handles=lines)
    ax.set_xlabel("Temperature (K)", fontsize=15)
    ax.set_ylabel("Nitrogen Conversion", fontsize=15)
    ax.set_title("Haber Process Conversion", fontsize=15)
    ax.set_xticklabels(np.round(ax.get_xticks(), decimals=2), fontsize=15)
    ax.set_yticklabels(np.round(ax.get_yticks(), decimals=2), fontsize=15)
    plt.savefig('conversion_lines.svg', format='svg', bbox_inches='tight')
else:  # Plot heatmap
    temp_mesh, pressure_mesh = np.meshgrid(temp_range, pressure_range)

    conversion_array = np.squeeze(np.array(conversion_solutions))
    fig, ax = plt.subplots()

    heatmap = ax.pcolormesh(temp_mesh, pressure_mesh, conversion_array, cmap="inferno")

    cbar = plt.colorbar(heatmap)
    cbar.set_label("Nitrogen Conversion")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (atm)")

    plt.savefig("conversion_heatmap.svg", format="svg", bbox_inches="tight")