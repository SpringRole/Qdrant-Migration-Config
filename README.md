Qdrant Config Migration Script-

1. In the folder create a file in the following format qdrant*migrator/migrations/<index>*<message>.py . Example qdrant_migrator/migrations/001_initial_migration.py . Now for the next version create 002
2. Now these file executions are version controlled and you can see the versions in GUI ->
   a. Go to collections there you will find migrations with a single point, with a metadata version. As of now it is in version 1 both dev and prod.
   b. Once you run the next version 002, it will change.
   c. You can't rerun versions forward. You have to create a new file with a new script to run next version.
3. There is a rollback feature also, (backward)-
   a. Your task will be to update the file with the backward function, for example, if you are writing something in v2, like create collection, hence it's rollback will be delete collection. Every file should consist of rollbacks also.
   b. There can be some non rollback files also, like I was not able to find methods to remove payload*indexes in v1. Hence my rollback function is empty. You can anytime come in and re-write the rollback method in future before rolling back.
   c. Ref code will be written below for file structure of qdrant_migrator/migrations/<index>*<message>.py
4. How to run the code? There are 2 clis -
   a. python cli.py migrate --url http://your-qdrant-url --api-key your-api-key --migration-folder migrations
   b. python cli.py rollback --url http://your-qdrant-url --api-key your-api-key --migration-folder migrations --target-version <target_version>

Summing it up, steps to go forward ->

1. Create a version file (ref pt.1).
2. Write the code for migration according to 3.c.
3. Execute 4.a.
4. Your Migration will be successfully completed.

For Backward/Rollback ->

1. Check its backward() function is written as expected.
2. Execute 4.b.
3. Your Rollback will be successfully completed.
4. You can track the current version from migration collections

Eg- code for point 3.c.-

```python
#Ensure each migration file has both `forward` and `backward` functions
def forward(client):
    # Your migration logic here
    client.create_collection('new_collection')
def backward(client):
    # Logic to undo the migration if needed
    client.delete_collection('new_collection')
```

Why to use this?

Now we can manage the versions and executions of scripts that we do run over. We don't have a GUI to change configs for qdrant, we have to do it via notebook. Now the thing was, we were not able to manage it across that who and what is getting updated in the config. This script will make it easy, we don't anymore need to create notebooks and, you need API keys and URL to execute any changes on collection, adding up to access control. Plus, we can commit the config changes over org repo, so, we will be able to control the versions too.
