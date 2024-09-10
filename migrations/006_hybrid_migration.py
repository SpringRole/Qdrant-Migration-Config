import os
import time
from qdrant_client import models, QdrantClient
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Import tqdm for progress bar
from fastembed import SparseTextEmbedding
import copy
import json
from qdrant_client.http.models import SparseVector

# Logging configuration
LOG_FILE = "migration.log"
# Clear existing handlers
logger = logging.getLogger(__name__)
logger.handlers = []

# Create handlers
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.CRITICAL)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

NEW_COLLECTION_NAME = "dobby-be-springworks-hybrid-collection"
SOURCE_COLLECTION_NAME = "dobby-be-springworks-collection-sharded"
OFFSET_FILE = "migration_offset.json"  # File to store the last offset
RETRY_DELAY = 5  # Delay (in seconds) before restarting after failure


def create_hybrid_collection(
    client: QdrantClient,
    new_collection_name: str,
    shard_number: int = 6,
    replication_factor: int = 2,
):
    client.create_collection(
        collection_name=new_collection_name,
        vectors_config={
            "text-dense": models.VectorParams(
                size=3072, distance=models.Distance.COSINE, on_disk=True
            )
        },
        sparse_vectors_config={
            "text-sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=True,
                )
            )
        },
        hnsw_config=models.HnswConfigDiff(
            payload_m=16,
            m=0,
        ),
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                always_ram=False,
            ),
        ),
        shard_number=shard_number,
        replication_factor=replication_factor,
    )


def add_payload_indexes(
    client: QdrantClient,
    collection_name: str,
    indices: list = [
        "organisation_id",
        "document_id",
        "ref_doc_id",
        "doc_id",
        "provider",
        "category",
        "res_name",
        "chunk_hash",
        "version",
    ],
):
    for index in indices:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=index,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def load_last_offset():
    """Load the last offset from the file."""
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE, "r") as f:
            return json.load(f).get("next_offset")
    return None


def save_last_offset(next_offset):
    """Save the last offset to the file."""
    with open(OFFSET_FILE, "w") as f:
        json.dump({"next_offset": next_offset}, f)


def generate_and_add_sparse_vector_to_records(records):
    SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    updated_records = []

    for record in records:
        node_content = record.payload.get("_node_content")
        if not node_content:
            continue
        parsed_content = json.loads(node_content)
        text = parsed_content.get("text", "")
        sparse_embedding = list(sparse_model.embed(str(text)))

        updated_record = copy.deepcopy(record)

        updated_record.vector = {
            "text-dense": record.vector,
            "text-sparse": SparseVector(
                indices=sparse_embedding[0].indices.tolist(),
                values=sparse_embedding[0].values.tolist(),
            ),
        }

        updated_records.append(updated_record)

    return updated_records


def migrate_points(
    client: QdrantClient,
    src_collection_name: str,
    dest_collection_name: str,
    batch_size: int = 100,
    max_workers: int = 4,  # Number of parallel workers
) -> None:
    def upload_batch(records, next_offset):
        try:
            if not records:
                logger.info("No records to upload")
                return
            updated_records = generate_and_add_sparse_vector_to_records(records)
            client.upload_records(dest_collection_name, updated_records, wait=False)
            logger.info(f"Moved {len(records)} records. {next_offset=}")
            save_last_offset(next_offset)  # Save offset after successful upload
            progress_bar.update(len(records))  # Update progress bar
        except Exception as e:
            logger.error(f"Failed to upload records batch: {e}")
            raise e

    # Get total number of poins in the source collection for the progress bar
    source_client_points_count = client.get_collection(src_collection_name).points_count
    logger.info(f"Total records to migrate: {source_client_points_count}")

    # Initialize the progress bar
    global progress_bar
    progress_bar = tqdm(
        total=source_client_points_count, desc="Migrating records", unit="records"
    )

    resume_offset = load_last_offset()
    if resume_offset:
        logger.info(f"Resuming migration from offset: {resume_offset}")
        records, next_offset = client.scroll(
            src_collection_name, limit=1, with_vectors=True, offset=resume_offset
        )
    else:
        records, next_offset = client.scroll(
            src_collection_name, limit=1, with_vectors=True
        )
    upload_batch(records, next_offset)
    logger.info("Migration started")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        while next_offset is not None:
            records, next_offset = client.scroll(
                src_collection_name,
                offset=next_offset,
                limit=batch_size,
                with_vectors=True,
            )
            if records:
                futures.append(executor.submit(upload_batch, records, next_offset))

        # Ensure all futures are completed
        for future in as_completed(futures):
            try:
                future.result()  # This will raise an exception if the future has one
            except Exception as e:
                logger.error(f"Exception in future: {e}")
                raise e

    dest_client_points_count = client.get_collection(dest_collection_name).points_count

    assert (
        source_client_points_count == dest_client_points_count
    ), f"Migration failed, points count are not equal: source vector count {source_client_points_count}, dest vector count {dest_client_points_count}"

    logger.info("Migration completed")
    progress_bar.close()  # Close progress bar when migration is complete


def run_migration_with_auto_restart(
    client, retries=float("inf"), retry_delay=RETRY_DELAY
):
    """Runs migration and automatically restarts if there's a failure."""
    attempt = 0
    while attempt < retries:
        try:
            migrate_points(
                client=client,
                src_collection_name=SOURCE_COLLECTION_NAME,
                dest_collection_name=NEW_COLLECTION_NAME,
            )
            break  # If migration completes successfully, exit the loop
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            attempt += 1
            logger.info(
                f"Retrying migration after {retry_delay} seconds (Attempt {attempt})..."
            )
            time.sleep(retry_delay)  # Wait before retrying


def forward(client):
    # create_hybrid_collection(client=client, new_collection_name=NEW_COLLECTION_NAME)
    # add_payload_indexes(client=client, collection_name=NEW_COLLECTION_NAME)
    run_migration_with_auto_restart(client)


def backward(client):
    # client.delete_collection(NEW_COLLECTION_NAME)
    print("Your code to rollback the migration here")
