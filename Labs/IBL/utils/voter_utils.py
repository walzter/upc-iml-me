# Temp placeholder impl. TODO by Andrey
from collections import Counter

def voter(dist_df, k):
  e_classes = dist_df.sort_values('euclidean_distance').head(k)['class'].values
  c_e = Counter(e_classes).most_common(1)[0][0]

  m_classes = dist_df.sort_values('manhattan_distance').head(k)['class'].values
  c_m = Counter(m_classes).most_common(1)[0][0]

  c_classes = dist_df.sort_values('clark_distance').head(k)['class'].values
  c_c = Counter(c_classes).most_common(1)[0][0]

  h_classes = dist_df.sort_values('hvdm_distance').head(k)['class'].values
  c_h = Counter(h_classes).most_common(1)[0][0]
  return c_e, c_m, c_c, c_h