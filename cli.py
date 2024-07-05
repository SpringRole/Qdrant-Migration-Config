import click
from migrator import migrate, rollback


@click.group()
def cli():
    pass


@click.command()
@click.option("--url", prompt="Qdrant URL", help="The URL of the Qdrant instance.")
@click.option(
    "--api-key", prompt="API Key", help="The API key for the Qdrant instance."
)
@click.option(
    "--migration-folder",
    prompt="Migration Folder",
    help="The folder containing migration scripts.",
)
def migrate_cmd(url, api_key, migration_folder):
    migrate(url, api_key, migration_folder)


@click.command()
@click.option("--url", prompt="Qdrant URL", help="The URL of the Qdrant instance.")
@click.option(
    "--api-key", prompt="API Key", help="The API key for the Qdrant instance."
)
@click.option(
    "--migration-folder",
    prompt="Migration Folder",
    help="The folder containing migration scripts.",
)
@click.option(
    "--target-version",
    prompt="Target Version",
    type=int,
    help="The target version to rollback to.",
)
def rollback_cmd(url, api_key, migration_folder, target_version):
    rollback(url, api_key, migration_folder, target_version)


cli.add_command(migrate_cmd, name="migrate")
cli.add_command(rollback_cmd, name="rollback")
if __name__ == "__main__":
    cli()
