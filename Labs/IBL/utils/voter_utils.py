from collections import Counter

def voter_most_voted(dist_df, k):
  e_classes = dist_df.sort_values('euclidean_distance').head(k)['class'].values
  c_e = Counter(e_classes).most_common(1)[0][0]

  m_classes = dist_df.sort_values('manhattan_distance').head(k)['class'].values
  c_m = Counter(m_classes).most_common(1)[0][0]

  c_classes = dist_df.sort_values('clark_distance').head(k)['class'].values
  c_c = Counter(c_classes).most_common(1)[0][0]

  h_classes = dist_df.sort_values('hvdm_distance').head(k)['class'].values
  c_h = Counter(h_classes).most_common(1)[0][0]
  return c_e, c_m, c_c, c_h

def voter_modified_plurality(dist_df, k):
  return compute_modified_plurality(dist_df, k, 'euclidean_distance'), compute_modified_plurality(dist_df, k, 'manhattan_distance'), compute_modified_plurality(dist_df, k, 'clark_distance'), compute_modified_plurality(dist_df, k, 'hvdm_distance')

def compute_modified_plurality(dist_df, k, distance_metric):
  k_classes = dist_df.sort_values(distance_metric).head(k)['class'].values
  sorted_count = sorted(Counter(k_classes).items(), key=lambda x: x[1], reverse=True)
  if (k > 1 and len(sorted_count) > 1 and sorted_count[0][1] == sorted_count[1][1]):
    return compute_modified_plurality(dist_df, k-1, distance_metric)
  else:
    return sorted_count[0][0]

def voter_borda_count(dist_df, k):
  e_classes = dist_df.sort_values('euclidean_distance').head(k)['class'].values
  m_classes = dist_df.sort_values('manhattan_distance').head(k)['class'].values
  c_classes = dist_df.sort_values('clark_distance').head(k)['class'].values
  h_classes = dist_df.sort_values('hvdm_distance').head(k)['class'].values

  e_classes_dict = {key: 0 for key in e_classes}
  m_classes_dict = {key: 0 for key in m_classes}
  c_classes_dict = {key: 0 for key in c_classes}
  h_classes_dict = {key: 0 for key in h_classes}

  for i in range(0, k):
    # Assign points (k-i) to ith element
    e_classes_dict.update({e_classes[i]: (k - i) + e_classes_dict.get(e_classes[i])})
    m_classes_dict.update({m_classes[i]: (k - i) + m_classes_dict.get(m_classes[i])})
    c_classes_dict.update({c_classes[i]: (k - i) + c_classes_dict.get(c_classes[i])})
    h_classes_dict.update({h_classes[i]: (k - i) + h_classes_dict.get(h_classes[i])})

  # Sort by the scores (values in the dict) & take the one with the highest points - winner in Borda count protocol
  c_e = sorted(e_classes_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
  c_m = sorted(m_classes_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
  c_c = sorted(c_classes_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
  c_h = sorted(h_classes_dict.items(), key=lambda x: x[1], reverse=True)[0][0]

  return c_e, c_m, c_c, c_h
