import matplotlib.pyplot as plt
import numpy as np

# Create a square figure to ensure the circle remains perfectly round
fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
ax.set_facecolor('black')

# Generate a nice oscilloscope-style waveform
x = np.linspace(2, 8, 1000)
y = (np.sin(2 * np.pi * x) * 1.2 + 
     0.6 * np.sin(4 * np.pi * x) + 
     0.3 * np.sin(8 * np.pi * x))

# Plot the glowing green waveform
ax.plot(x, y, color='#00FF00', linewidth=3.5, alpha=0.95)

# Add a subtle glow effect
for i in range(3, 0, -1):
    ax.plot(x, y, color='#00FF00', linewidth=3.5 + i*2.5, alpha=0.12/i)

# Add a faint grid like an oscilloscope
ax.grid(True, color='#003300', linestyle='-', linewidth=0.8, alpha=0.6)

# Set equal aspect ratio and limits so the circle is perfectly round
ax.set_xlim(0, 10)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')  # This forces the circle to be a true circle, not oval

# Draw the outer circle (perfectly round)
circle = plt.Circle((5, 0), 2.25, fill=False, edgecolor='#00FF00', linewidth=4, alpha=0.9)
ax.add_patch(circle)

# Remove axes for clean oscilloscope look
ax.axis('off')

# Tight layout and save
plt.tight_layout()
plt.savefig('siglab_icon.png', dpi=300, bbox_inches='tight', facecolor='black')
plt.show()