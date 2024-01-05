import numpy as np
from matplotlib import pyplot as plt

# Heat capacity coefficients
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

DH_STD = -45.4 * 1000  # J/mol
STD_TEMP = 298  # Kelvin
N_N2 = 1  # Initial moles of N2
P_STD = 1  # atm


def integrate_heat_cap(A, B, C, D, E, initial_temp, final_temp):
    t_1 = initial_temp / 1000
    t_2 = final_temp / 1000

    term_1 = (A * t_1) + (B * t_1 ** 2 / 2) + (C * t_1 ** 3 / 3) + (D * t_1 ** 4 / 4) - (E / t_1)
    term_2 = (A * t_2) + (B * t_2 ** 2 / 2) + (C * t_2 ** 3 / 3) + (D * t_2 ** 4 / 4) - (E / t_2)

    return term_1 - term_2


PHASE_SPACE = 100  # Side length of square phase space

temp_range = np.linspace(298, 900, PHASE_SPACE)  # In Kelvin
pressure_range = np.linspace(10, 300, PHASE_SPACE)  # In atm

conversion_solutions = []

for pressure in pressure_range:
    pressure_solution = []

    for temperature in temp_range:
        nitrogen_heat = integrate_heat_cap(initial_temp=temperature, final_temp=STD_TEMP, **N2_COEFFS_2000K)
        hydrogen_heat = integrate_heat_cap(initial_temp=temperature, final_temp=STD_TEMP, **H2_COEFFS)
        ammonia_heat = integrate_heat_cap(initial_temp=STD_TEMP, final_temp=temperature, **NH3_COEFFS)

        # Enthalpy of reaction at desired temperature
        rxn_enthalpy = DH_STD + nitrogen_heat + 3 * hydrogen_heat + 2 * ammonia_heat

        vanthoff_term = (rxn_enthalpy / 8.314) * (1 / STD_TEMP - 1 / temperature)

        exp_term = np.sqrt(np.exp(vanthoff_term))

        m = (9 / 2) * exp_term * (pressure / P_STD)

        nitrogen_reacted = 1 - (np.sqrt(2)) / (np.sqrt(m + 2))
        conversion = nitrogen_reacted / N_N2
        pressure_solution.append(conversion)

    conversion_solutions.append(pressure_solution)

# Creat heat map 
fig, ax = plt.subplots(figsize=[8, 6])

temp_mesh, pressure_mesh = np.meshgrid(temp_range, pressure_range)

conversion_array = np.squeeze(np.array(conversion_solutions))

heatmap = ax.pcolormesh(temp_mesh, pressure_mesh, conversion_array, cmap="inferno")

cbar = plt.colorbar(heatmap)
cbar.set_label("Nitrogen Conversion")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Pressure (atm)")

plt.savefig("conversion_heatmap.svg", format="svg", bbox_inches="tight")