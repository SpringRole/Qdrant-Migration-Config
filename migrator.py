import os
import importlib.util
from qdrant_client import QdrantClient, models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def initialize_qdrant_client(url, api_key):
    client = QdrantClient(url=url, api_key=api_key, prefer_grpc=True)
    return client


def check_and_create_migrations_collection(client):
    collections_response = client.get_collections()
    collections = collections_response.collections
    collection_names = [collection.name for collection in collections]
    if "migrations" not in collection_names:
        client.create_collection(
            collection_name="migrations",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        )
        client.upsert(
            collection_name="migrations",
            points=[
                models.PointStruct(
                    id="5c56c793-69f3-4fbf-87e6-c4bf54c28c26",
                    payload={
                        "version": 0,
                    },
                    vector=[0.0, 0.1],
                ),
            ],
        )


def get_current_version(client):
    points = client.scroll(
        "migrations",
        with_payload=True,
    )
    if points:
        version = points[0][0].payload["version"]
        return version
    return 0


def set_current_version(client, version):
    client.upsert(
        collection_name="migrations",
        points=[
            models.PointStruct(
                id="5c56c793-69f3-4fbf-87e6-c4bf54c28c26",
                payload={
                    "version": version,
                },
                vector=[0.0, 0.1],
            ),
        ],
    )


def get_migration_files(migration_folder):
    files = []
    for file in os.listdir(migration_folder):
        if file.endswith(".py"):
            index = int(file.split("_")[0])
            files.append((index, file))
    files.sort(key=lambda x: x[0])
    return files


def run_migrations(client, migration_folder, current_version, target_version=None):
    migration_files = get_migration_files(migration_folder)
    for index, file in migration_files:
        if index > current_version and (
            target_version is None or index <= target_version
        ):
            module_name = file.replace(".py", "")
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(migration_folder, file)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.forward(client)
            set_current_version(client, index)
            logger.info(
                f"Migration completed successfully for {file}! Enjoy your migration :D"
            )
        else:
            logger.info(f"Skipping migration {file}")


def rollback_migrations(client, migration_folder, current_version, target_version):
    migration_files = get_migration_files(migration_folder)
    for index, file in reversed(migration_files):
        if index <= current_version and index > target_version:
            module_name = file.replace(".py", "")
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(migration_folder, file)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.backward(client)
            set_current_version(client, index - 1)
            logger.info(
                f"Rollback completed successfully for {file}! Enjoy your migration :D"
            )
        else:
            logger.info(f"Skipping rollback {file}")


def migrate(url, api_key, migration_folder):
    client = initialize_qdrant_client(url, api_key)
    check_and_create_migrations_collection(client)
    current_version = get_current_version(client)
    run_migrations(client, migration_folder, current_version)


def rollback(url, api_key, migration_folder, target_version):
    client = initialize_qdrant_client(url, api_key)
    check_and_create_migrations_collection(client)
    current_version = get_current_version(client)
    rollback_migrations(client, migration_folder, current_version, target_version)
