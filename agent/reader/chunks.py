def split_text(text, size=6000):

    chunks = []

    for i in range(0, len(text), size):
        chunks.append(text[i:i + size])

    return chunks
