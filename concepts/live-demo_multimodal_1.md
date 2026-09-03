# Live Demo Guide - Introduction to Multimodal AI


### Install

```bash
pip install sentence-transformers torch scikit-image pillow pandas numpy matplotlib scikit-learn
```

### Pre-cache the embedding model


```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("clip-ViT-B-32")
print("Model cached")
PY
```

### Verify everything loads

```python
from sentence_transformers import SentenceTransformer
from skimage import data
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt

model = SentenceTransformer("clip-ViT-B-32")
img = Image.fromarray(data.chelsea())
print("ok", img.size)
```

### Offline fallback

The sample images are bundled with `scikit-image`. The only major network dependency is the first model download. Use a pre-cached environment if classroom WiFi is unreliable.


## Step-by-Step Commands

### Step 1 - Load and display sample images

```python
from skimage import data
from PIL import Image
import matplotlib.pyplot as plt

images = {
    "cat": Image.fromarray(data.chelsea()),
    "coffee": Image.fromarray(data.coffee()),
    "rocket": Image.fromarray(data.rocket()),
    "astronaut": Image.fromarray(data.astronaut()),
}

plt.figure(figsize=(10, 3))
for i, (name, img) in enumerate(images.items(), start=1):
    plt.subplot(1, len(images), i)
    plt.imshow(img)
    plt.title(name)
    plt.axis("off")

plt.tight_layout()
plt.show()
```


### Step 2 - Load the model and define descriptions

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("clip-ViT-B-32")
print("Model loaded")
```

```python
image_names = list(images.keys())
image_values = list(images.values())

descriptions = [
    "a cat sitting indoors",
    "a cup of coffee on a table",
    "a rocket in the sky",
    "an astronaut in a space suit",
]

for i, text in enumerate(descriptions, start=1):
    print(i, text)
```

### Step 3 - Generate embeddings

```python
text_embeddings = model.encode(
    descriptions,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

image_embeddings = model.encode(
    image_values,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print("Text embeddings:", text_embeddings.shape)
print("Image embeddings:", image_embeddings.shape)
print("First 10 numbers:", text_embeddings[0][:10])
```


### Step 4 - Compute cosine similarity

```python
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

scores = cosine_similarity(image_embeddings, text_embeddings)

similarity_table = pd.DataFrame(
    scores,
    index=image_names,
    columns=descriptions,
)

similarity_table.round(3)
```



### Step 5 - Match each image to the best description

```python
for image_idx, image_name in enumerate(image_names):
    best_text_idx = scores[image_idx].argmax()
    best_description = descriptions[best_text_idx]
    best_score = scores[image_idx, best_text_idx]

    print(f"{image_name:10s} -> {best_description} ({best_score:.3f})")
```

Expected:

* `cat` should match `a cat sitting indoors`
* `coffee` should match `a cup of coffee on a table`
* `rocket` and `astronaut` should score strongly on space-related descriptions



### Step 6 - Search images with a text query

```python
query = "something related to space travel"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True,
)

query_scores = cosine_similarity(query_embedding, image_embeddings)[0]

ranked_results = pd.DataFrame({
    "image": image_names,
    "score": query_scores,
}).sort_values("score", ascending=False)

ranked_results
```



### Step 7 - One ambiguous query

```python
query = "a person in a special outfit"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True,
)

query_scores = cosine_similarity(query_embedding, image_embeddings)[0]

pd.DataFrame({
    "image": image_names,
    "score": query_scores,
}).sort_values("score", ascending=False)
```




## Key Instructor Talking Points

* A modality is a type of input, such as text, image, audio, or video.
* Encoders convert raw inputs into vectors.
* Embeddings are numerical representations, not explanations.
* Shared embedding spaces allow cross-modal comparison.
* Cosine similarity supports retrieval, but it does not prove truth.
* Retrieval is not the same as reasoning.

---

## Challenge for Students

Try these queries and see what happens:

**Challenge 1 - Describe an image**
Write your own description for one of the images and see how high the score gets.

**Challenge 2 - Ambiguous query**
Create a query that could match multiple images. Example: `"something round"` or `"something that flies"`. Which image gets picked? Why?

**Challenge 3 - Add a new image**
(Optional) Add your own image to the search. What descriptions work best? What confuses the model?

**Key question to discuss:**
If the model gives a high score, does that mean the match is correct? Why or why not?

---

## Suggested Solutions

### Solution for Challenge 1

```python
# Try different descriptions for the cat image
test_descriptions = [
    "a cat sitting indoors",      # Original - should score high
    "a tabby cat",                # More specific - should still score well
    "an orange animal",           # Vague - might score lower
    "a fluffy creature",          # Generic - probably lower
]

test_embeddings = model.encode(test_descriptions, convert_to_numpy=True, normalize_embeddings=True)
test_scores = cosine_similarity([image_embeddings[0]], test_embeddings)[0]  # Compare with cat image

for desc, score in zip(test_descriptions, test_scores):
    print(f"{score:.3f} -> {desc}")
```

**What happens:** More specific descriptions usually score higher because they match the model's training better.

---

### Solution for Challenge 2

```python
# Test an ambiguous query
ambiguous_query = "something round"

query_embedding = model.encode([ambiguous_query], convert_to_numpy=True, normalize_embeddings=True)
query_scores = cosine_similarity(query_embedding, image_embeddings)[0]

pd.DataFrame({
    "image": image_names,
    "score": query_scores,
}).sort_values("score", ascending=False)
```

**What happens:** The model might pick the coffee cup (because cups are round) or even the rocket (because it's cylindrical). This shows how similarity can be ambiguous!

---

### Solution for Challenge 3

```python
# Load a new image and add it to the search
from PIL import Image

# Example: load a new image (replace path with your image)
new_image = Image.open("path/to/your/image.jpg")

# Add to our images
new_images = list(image_values) + [new_image]
new_names = list(image_names) + ["my_image"]

# Re-encode everything
new_image_embeddings = model.encode(new_images, convert_to_numpy=True, normalize_embeddings=True)

# Test a query
query = "describe your image here"
query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
query_scores = cosine_similarity(query_embedding, new_image_embeddings)[0]

pd.DataFrame({
    "image": new_names,
    "score": query_scores,
}).sort_values("score", ascending=False)
```

**What happens:** See how well the model understands your custom image!

---

### Discussion: High Score ≠ Correct

**The answer:** No, a high score does NOT guarantee correctness.

**Why?**
* The model learned patterns, not facts
* Similar patterns can come from different things
* The model might miss small but important details
* Ambiguous queries can match the wrong image

**Example from the demo:**
* "a person in a special outfit" matches both astronaut AND the cat (false positive!)
* Just because the score is high doesn't mean it's the right match

**Key takeaway:** Always inspect the results, not just the scores.

---

