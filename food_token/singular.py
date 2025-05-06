def clean_ingredients(file_path, cleaned_path, pairs_path):

    def candidate_singulars(word):
        # Generate possible singular forms from a plural
        candidates = set()
        if word.endswith('ies') and len(word) > 3:
            candidates.add(word[:-3] + 'y')  # berries → berry
        elif word.endswith('es') and len(word) > 2:
            candidates.add(word[:-2])        # tomatoes → tomato
            candidates.add(word[:-1])        # apples → apple (try both)
        elif word.endswith('s') and not word.endswith(('ss', 'us')) and len(word) > 1:
            candidates.add(word[:-1])        # lentils → lentil
        return candidates

    with open(file_path, 'r', encoding='utf-8') as f:
        original_lines = [line.strip().lower() for line in f if line.strip()]

    unique_words = set(original_lines)
    cleaned = set(unique_words)
    plural_mapping = {}

    for word in unique_words:
        for singular in candidate_singulars(word):
            if singular in unique_words and singular != word:
                if word in cleaned:
                    cleaned.remove(word)
                plural_mapping[word] = singular
                break  # stop at first matching singular

    # Write cleaned file
    with open(cleaned_path, 'w', encoding='utf-8') as f:
        for item in sorted(cleaned):
            f.write(item + '\n')

    # Write plural-to-singular map
    with open(pairs_path, 'w', encoding='utf-8') as f:
        for plural, singular in sorted(plural_mapping.items()):
            f.write(f"{plural} -> {singular}\n")

    print(f"Cleaned list written to: {cleaned_path}")
    print(f"Plural pairs written to: {pairs_path}")



clean_ingredients('1_filtered.txt', '1_cleaned.txt', "plural_pair.txt")

