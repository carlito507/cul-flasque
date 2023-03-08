import logging
import sys
import docx2txt

# from PyPDF2 import PdfReader
from PyPDF3 import PdfFileReader
from numpy import array, average
from flask import current_app
from config import *

from utils import get_embeddings, get_pinecone_id_for_file_chunk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


def handle_file(file, session_id, pinecone_index, tokenizer):
    filename = file.filename
    logging.info("[handle_file] Handling file: {}".format(filename))
    file_text_dict = current_app.config["file_text_dict"]
    try:
        extracted_text = extract_text_from_file(file)
    except ValueError as e:
        logging.error(
            "[handle_file] Error extracting text from file: {}".format(e))
        raise e
    file_text_dict[filename] = extracted_text
    return handle_file_string(filename, session_id, extracted_text, pinecone_index, tokenizer, file_text_dict)


def extract_text_from_file(file):
    extracted_text = ""
    if file.mimetype == "application/pdf":
        reader = PdfFileReader(file)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extractText()
    elif file.mimetype == "text/plain":
        extracted_text = file.read().decode("utf-8")
        file.close()
    elif file.mimetype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        extracted_text = docx2txt.process(file)
    return extracted_text


def handle_file_string(filename, session_id, extracted_text, pinecone_index, tokenizer, file_text_dict):
    logging.info("[handle_file_string] Handling file string: {}".format(filename))
    # Tokenize the text
    tokenized_text = tokenizer(extracted_text)
    # Get the embeddings for the text
    embeddings = get_embeddings(tokenized_text, ENGINE)
    # Get the average embedding for the text
    average_embedding = average(array(embeddings), axis=0).tolist()
    # Get the pinecone id for the file
    pinecone_id = get_pinecone_id_for_file_chunk(session_id, filename, 0)
    # Upsert the average embedding to Pinecone
    pinecone_index.upsert(pinecone_id, average_embedding)
    # Return the file text dict
    return file_text_dict


def handle_file_string(filename, session_id, extracted_text, pinecone_index, tokenizer, file_text_dict):
    logging.info("[handle_file_string] Handling file string: {}".format(filename))
    clean_file_body_string = file_body_string.replace(
        "\n", "; ").replace("  ", " ")
    text_to_embed = "Filename is: {}; {}".format(
        filename, clean_file_body_string)

    # Create embeddings for the text
    try:
        text_embeddings, average_embedding = create_embeddings_for_text(
            text_to_embed, tokenizer)
        logging.info(
            "[handle_file_string] Created embedding for {}".format(filename))
    except Exception as e:
        logging.error(
            "[handle_file_string] Error creating embedding: {}".format(e))
        raise e

    # Get the vectors array of triples: file_chunk_id, embedding, metadata for each embedding
    # Metadata is a dict with keys: filename, file_chunk_index
    vectors = []
    for i, (text_chunk, embedding) in enumerate(text_embeddings):
        id = get_pinecone_id_for_file_chunk(session_id, filename, i)
        file_text_dict[id] = text_chunk
        vectors.append(
            (id, embedding, {"filename": filename, "file_chunk_index": i}))

        logging.info(
            "[handle_file_string] Text chunk {}: {}".format(i, text_chunk))

    # Split the vectors array into smaller batches of max length 2000
    batch_size = MAX_PINECONE_VECTORS_TO_UPSERT_PATCH_SIZE
    batches = [vectors[i:i+batch_size] for i in range(0, len(vectors), batch_size)]

    # Upsert each batch to Pinecone
    for batch in batches:
        try:
            pinecone_index.upsert(
                vectors=batch, namespace=session_id)

            logging.info(
                "[handle_file_string] Upserted batch of embeddings for {}".format(filename))
        except Exception as e:
            logging.error(
                "[handle_file_string] Error upserting batch of embeddings to Pinecone: {}".format(e))
            raise e


def get_col_average_from_list_of_lists(list_of_lists):
    if len(list_of_lists) == 1:
        return list_of_lists[0]
    else:
        list_of_lists_array = array(list_of_lists)
        average_embedding = average(list_of_lists_array, axis=0)
        return average_embedding.tolist()


def create_embeddings_for_text(text_to_embed, tokenizer):
    # Tokenize the text
    token_chunks = list(chunks(text_to_embed, TEXT_EMBEDDING_CHUNK_SIZE, tokenizer))
    text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]
    text_chunks_array = [text_chunks[i:i+MAX_TEXTS_TO_EMBED_BATCH_SIZE] for i in range(0, len(text_chunks), MAX_TEXTS_TO_EMBED_BATCH_SIZE)]
    embeddings = []
    for text_chunks_array in text_chunks_arrays:
        embeddings_response = get_embeddings(text_chunks_array, EMBEDDINGS_MODEL)
        embeddings.extend([embedding["embedding"] for embedding in embeddings_response])

    text_embeddings = list(zip(text_chunks, embeddings))

    average_embedding = get_col_average_from_list_of_lists(embeddings)

    return (text_embeddings, average_embedding)


def chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j