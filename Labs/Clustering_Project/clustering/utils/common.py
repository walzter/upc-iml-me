def list_to_numbers(set_labels, target_list_pattern):
    final_list = []
    for e in target_list_pattern:
        for c, s in enumerate(set_labels):
            if e == s:
                final_list.append(c)
                break
    return final_list