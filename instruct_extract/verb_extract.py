import spacy
import json
import multiprocessing
from collections import Counter
import time

# Shared globals
counter = None
lock = None
shared_freq = None
start_time = None  # for timing

def init_worker(shared_counter, shared_lock, shared_freq_dict, shared_start_time):
    global counter, lock, shared_freq, start_time
    counter = shared_counter
    lock = shared_lock
    shared_freq = shared_freq_dict
    start_time = shared_start_time

def extract_verbs_batch(recipe_batch):
    nlp = spacy.load("en_core_web_trf")
    local_counter = Counter()
    for recipe in recipe_batch:
        instructions = recipe.get("instructions", [])
        valid_instructions = [instr for instr in instructions if isinstance(instr, str) and instr.strip()]
        for doc in nlp.pipe(valid_instructions, batch_size=32):
            for token in doc:
                if token.pos_ == "VERB":
                    local_counter[token.lemma_] += 1
        with lock:
            counter.value += 1
            for verb, freq in local_counter.items():
                shared_freq[verb] = shared_freq.get(verb, 0) + freq
            if counter.value % 100 == 0:
                elapsed = time.time() - start_time.value
                print(f"✅ Processed {counter.value} recipes in {elapsed:.2f} seconds")
                save_outputs(shared_freq)
        local_counter.clear()
    return

def save_outputs(freq_dict):
    sorted_freq = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    with open("verb_frequencies.txt", "w", encoding="utf-8") as f:
        for verb, freq in sorted_freq:
            f.write(f"{verb}: {freq}\n")
    with open("top_100_verbs.txt", "w", encoding="utf-8") as f:
        for verb, _ in sorted_freq[:2000]:
            f.write(verb + "\n")

if __name__ == "__main__":
    # Load recipes
    with open("../data/train_data.json", "r", encoding="utf-8") as f:
        recipes = json.load(f)

    # Divide recipes
    MAX_PROCESSES = 3
    chunk_size = len(recipes) // MAX_PROCESSES + 1
    chunks = [recipes[i:i + chunk_size] for i in range(0, len(recipes), chunk_size)]

    # Setup shared state
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    shared_freq = manager.dict()
    shared_start_time = manager.Value('d', time.time())  # shared float

    # Run multiprocessing
    with multiprocessing.Pool(
        processes=MAX_PROCESSES,
        initializer=init_worker,
        initargs=(counter, lock, shared_freq, shared_start_time)
    ) as pool:
        pool.map(extract_verbs_batch, chunks)

    # Final save
    save_outputs(shared_freq)

    print(f"\n🎉 Done. Extracted {len(shared_freq)} unique verbs from {len(recipes)} recipes.")
    print("📁 Saved: 'verb_frequencies.txt' and 'top_1000_verbs.txt'")





