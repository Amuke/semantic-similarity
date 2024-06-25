import spacy

# Load the medium language model
nlp_md = spacy.load('en_core_web_md')

# Compare single words
word1 = nlp_md("cat")
word2 = nlp_md("monkey")
word3 = nlp_md("banana")

print("Comparing single words with en_core_web_md:")
print(f"Similarity between '{word1}' and '{word2}': {word1.similarity(word2)}")
print(f"Similarity between '{word3}' and '{word2}': {word3.similarity(word2)}")
print(f"Similarity between '{word3}' and '{word1}': {word3.similarity(word1)}\n")

# Compare series of words
tokens = nlp_md('cat apple monkey banana')
print("Comparing series of words with en_core_web_md:")
for token1 in tokens:
    for token2 in tokens:
        print(f"{token1.text} - {token2.text}: {token1.similarity(token2)}")
print()

# Compare sentences
sentence_to_compare = "Why is my cat on the car"
sentences = [
    "where did my dog go",
    "Hello, there is my car",
    "I've lost my car in my car",
    "I'd like my boat back",
    "I will name my dog Diana"
]

model_sentence = nlp_md(sentence_to_compare)
print("Comparing sentences with en_core_web_md:")
for sentence in sentences:
    similarity = nlp_md(sentence).similarity(model_sentence)
    print(f"{sentence} - {similarity}")

# Note: Load the simpler language model
nlp_sm = spacy.load('en_core_web_sm')

# Compare single words
word1_sm = nlp_sm("cat")
word2_sm = nlp_sm("monkey")
word3_sm = nlp_sm("banana")

print("\nComparing single words with en_core_web_sm:")
print(f"Similarity between '{word1_sm}' and '{word2_sm}': {word1_sm.similarity(word2_sm)}")
print(f"Similarity between '{word3_sm}' and '{word2_sm}': {word3_sm.similarity(word2_sm)}")
print(f"Similarity between '{word3_sm}' and '{word1_sm}': {word3_sm.similarity(word1_sm)}\n")

# Compare series of words
tokens_sm = nlp_sm('cat apple monkey banana')
print("Comparing series of words with en_core_web_sm:")
for token1 in tokens_sm:
    for token2 in tokens_sm:
        print(f"{token1.text} - {token2.text}: {token1.similarity(token2)}")
print()

# Compare sentences
model_sentence_sm = nlp_sm(sentence_to_compare)
print("Comparing sentences with en_core_web_sm:")
for sentence in sentences:
    similarity = nlp_sm(sentence).similarity(model_sentence_sm)
    print(f"{sentence} - {similarity}")

# Notes:
# en_core_web_md captures more nuanced similarities due to its larger vocabulary and word vectors.
# en_core_web_sm lacks word vectors, leading to less accurate similarity scores.

# My own example:
word4 = nlp_md("apple")
word5 = nlp_md("fruit")
print(f"\nSimilarity between '{word4}' and '{word5}': {word4.similarity(word5)}")

