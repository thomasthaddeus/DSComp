# Compute the confusion matrix
conf_mat = confusion_matrix(val_labels, val_labels)
sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()