import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- STEP 1: DATA LOADING ---
# Loading the 3-variable dataset [cite: 8, 16]
try:
    file_path = 'clean.csv'
    df = pd.read_csv(file_path)
    print("✅ Data successfully loaded!")
    print("DataFrame columns:", list(df.columns))
except FileNotFoundError:
    print("❌ Error: 'ansys_results.csv' not found. Please place it in the same folder.")
    exit()


column_names = df.columns
assert 'InsulationThickness' in column_names, "Error: 'InsulationThickness' column missing."
assert 'HeatFlux' in column_names, "Error: 'HeatFlux' column missing."

# Assign variables based on updated parameter names [cite: 18, 19]
X = df[['InsulationThickness', 'HeatFlux']]  # Input variables
y = df['Tskin']                              # Target variable

# --- STEP 2: AI MODEL TRAINING & VALIDATION ---
# Splitting data to validate using a Test Set (Deliverable Requirement) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Regression Model: Tskin = f(InsulationThickness, HeatFlux) [cite: 22]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- STEP 3: ERROR ANALYSIS (For Technical Report) ---
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL PERFORMANCE ANALYSIS ---")
print(f"Dataset Size: {len(df)} cases")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} °C")
print(f"Model Accuracy (R²): {r2:.4f}")

# --- STEP 4: VISUALIZATION (DESIGN SPACE CONTOUR) ---
# Generating the contour plot for the technical report [cite: 24, 31]
thickness_range = np.linspace(1, 5, 100)
flux_range = np.linspace(0.1, 2.0, 100)
T_grid = np.zeros((100, 100))

# Filling the grid with AI predictions
for i, t_val in enumerate(thickness_range):
    # Using the new parameter names in the DataFrame to avoid warnings
    input_batch = pd.DataFrame({'InsulationThickness': [t_val] * 100, 'HeatFlux': flux_range})
    T_grid[:, i] = model.predict(input_batch)

plt.figure(figsize=(10, 7))
cp = plt.contourf(thickness_range, flux_range, T_grid, cmap='RdYlBu_r', levels=25)
plt.colorbar(cp, label='Skin Temperature ($T_{skin}$) [°C]')

# Safety Constraint: Must keep Tskin <= 43.5°C [cite: 26]
cs = plt.contour(thickness_range, flux_range, T_grid, levels=[43.5], colors='red', linewidths=3, linestyles='--')
plt.clabel(cs, inline=True, fontsize=12, fmt='SAFETY LIMIT (43.5°C)')

plt.xlabel('Insulation Thickness [mm]')
plt.ylabel('Device Heat Flux [W/(m^2)]')
plt.title('AI-Generated Design Space: Performance vs. Safety')
plt.grid(alpha=0.3)
plt.savefig('design_space_contour.png', dpi=300)
print("\n✅ Contour plot saved as 'design_space_contour.png'")

# --- STEP 5: OPTIMIZATION (FINDING MAX Q) ---
# Iteratively finding the combination that maximizes HeatFlux while staying safe [cite: 11, 26]
best_flux, best_thickness = 0, 0
FDA_RECOMMENDED_TEMP = 43.5  # Maximum safe skin temperature in °C [cite: 26]

for t_val in thickness_range:
    # Test heat flux from high to low to find the maximum safe limit
    for q_val in reversed(flux_range):
        pred_input = pd.DataFrame({'InsulationThickness': [t_val], 'HeatFlux': [q_val]})
        pred = model.predict(pred_input)[0]
        if pred <= FDA_RECOMMENDED_TEMP:
            if q_val > best_flux:
                best_flux, best_thickness = q_val, t_val
            break

print("\n--- FINAL RECOMMENDATION ---")
print(f"To maximize performance, we recommend an insulation thickness of {best_thickness:.2f} mm.")
print(f"This allows for a maximum power output of {best_flux:.2f} Watts while maintaining skin safety.")