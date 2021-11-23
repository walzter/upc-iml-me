for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = normalized_dataframe[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0],xy[:, 1],"o",markerfacecolor=tuple(col),
                                      markeredgecolor="k",
                                      markersize=14
                                      )

        xy = normalized_dataframe[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0],xy[:, 1],"o",markerfacecolor=tuple(col),
                                       markeredgecolor="k",
                                       markersize=6,
                                           )
