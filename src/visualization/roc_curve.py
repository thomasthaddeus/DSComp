# Compute the ROC curve and the AUC
fpr, tpr, thresholds = roc_curve(val_labels, val_labels)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(
    fpr, tpr, color="#FF8C00", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()