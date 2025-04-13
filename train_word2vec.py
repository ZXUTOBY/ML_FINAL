import pandas as pd
from gensim.models import Word2Vec
import ast

def load_and_preprocess(data_path: str) -> list:
    data = pd.read_csv(data_path)
    data["NER"] = data["NER"].apply(ast.literal_eval)  # Convert string to list
    return data["NER"].tolist()

def train_word2vec(corpus: list) -> Word2Vec:
    # Chen's setup
    return Word2Vec(
        sentences=corpus,
        vector_size=100, 
        window=5,        
        min_count=1,    
        epochs=50,        
        sg=1,            
        workers=4
    )

def save_model(model: Word2Vec, output_path: str) -> None:
    model.save(output_path)
    print(f"Model saved to {output_path}")

def validate_embeddings(model: Word2Vec):
    test_pairs = [("chicken", "butter"), ("garlic", "onion"), ("tomato", "basil")]
    print("\nEmbedding Validation:")
    for word1, word2 in test_pairs:
        try:
            similarity = model.wv.similarity(word1, word2)
            print(f"{word1} ↔ {word2}: {similarity:.2f}")
        except KeyError as e:
            print(f"Missing ingredient: {e}")

def main():
    data_path = "/Users/chrisdeng/Desktop/COSC410/final/ML_FINAL/20rows.csv"    # Modify Data Path
    output_path = "/Users/chrisdeng/Desktop/COSC410/final/ML_FINAL/recipe_embeddings.model" 

    # Load data
    corpus = load_and_preprocess(data_path)
    print(f"Loaded {len(corpus)} recipes")

    # Train model
    model = train_word2vec(corpus)
    
    # Save and validate
    save_model(model, output_path)
    validate_embeddings(model)

if __name__ == "__main__":
    main()