from matplotlib import colors


# CENTRAL METEOROLOGICAL REPRICING BOUNDS (dBZ/Rainfall)
rainfall_cmap = colors.ListedColormap([
    "silver", "white", "darkslateblue", "mediumblue", "dodgerblue", "skyblue",
    "olive", "mediumseagreen", "cyan", "lime", "yellow", "khaki", "burlywood",
    "orange", "brown", "pink", "red", "plum", "purple"
])

rainfall_bounds = [-1, 0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 75, 254]
rainfall_norm = colors.BoundaryNorm(rainfall_bounds, rainfall_cmap.N)

# ERROR ANALYSIS REPRICING BOUNDS (Model Deviations)
error_cmap = colors.ListedColormap([
    "#08306b", "#08519c", "#2171b5", "#6baed6", "#c6dbef", # Underprediction
    "#ffffff",                                             # Balanced
    "#fdd0c7", "#fca082", "#fb6a4a", "#de2d26", "#67000d"  # Overprediction
])
error_cmap.set_under("#000000")
error_cmap.set_over("#4d004b")

error_bounds = [-75, -15, -8, -4, -2, -1, 1, 2, 4, 8, 15, 75]
error_norm = colors.BoundaryNorm(error_bounds, error_cmap.N)
