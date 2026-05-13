import matplotlib.pyplot as plt
import numpy as np

# Data from evaluation.json (per-category semantic similarity means)
categories = [
    "Glaucoma",
    "High Blood Pressure",
    "Osteoarthritis",
    "Diabetes",
    "Alzheimer's Disease",
    "Urinary Tract Infections",
    "Hearing Loss",
    "Prostate Cancer",
    "Osteoporosis",
    "COPD",
    "Stroke",
    "Age-related Macular\nDegeneration",
    "Peripheral Arterial\nDisease (P.A.D.)",
    "Breast Cancer",
    "Dry Mouth",
    "Paget's Disease\nof Bone",
    "Anemia"
]

scores = [
    0.939,  # Glaucoma
    0.868,  # High Blood Pressure
    0.940,  # Osteoarthritis
    0.919,  # Diabetes
    0.952,  # Alzheimer's Disease
    0.923,  # Urinary Tract Infections
    0.953,  # Hearing Loss
    0.950,  # Prostate Cancer
    0.837,  # Osteoporosis
    0.954,  # COPD
    0.932,  # Stroke
    0.969,  # Age-related Macular Degeneration
    0.987,  # Peripheral Arterial Disease
    0.933,  # Breast Cancer
    0.945,  # Dry Mouth
    0.905,  # Paget's Disease of Bone
    0.926   # Anemia
]

# Create color gradient based on score (blue to green)
norm = plt.Normalize(min(scores), max(scores))
colors = plt.cm.RdYlGn(norm(scores))  # Red-Yellow-Green colormap

# Create figure
plt.figure(figsize=(14, 8))
bars = plt.bar(range(len(categories)), scores, color=colors, edgecolor='black', linewidth=0.5)

# Add horizontal failure threshold line
plt.axhline(y=0.6, color='red', linestyle='--', linewidth=2, label='Failure Threshold (0.6)')

# Add value labels on top of bars
for i, (bar, score) in enumerate(zip(bars, scores)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontsize=9)

# Labels and title
plt.xlabel('Focus Area', fontsize=12, fontweight='bold')
plt.ylabel('Semantic Similarity Score', fontsize=12, fontweight='bold')
plt.title('Semantic Similarity Scores by Test Case', fontsize=14, fontweight='bold')

# X-axis ticks
plt.xticks(range(len(categories)), categories, rotation=45, ha='right', fontsize=10)
plt.yticks(np.arange(0, 1.1, 0.1))

# Set y-axis limit
plt.ylim(0, 1.05)

# Add grid
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Add legend
plt.legend(loc='lower right')

# Add global mean annotation
global_mean = np.mean(scores)
plt.text(0.02, 0.98, f'Global Mean: {global_mean:.3f}', transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('semantic_similarity_bar_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('semantic_similarity_bar_chart.pdf', bbox_inches='tight')  # Vector format for LaTeX

# Show plot
plt.show()